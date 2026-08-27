# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, List, Optional, Sequence, Tuple
import torch
import os
from tqdm import tqdm

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
import torch.distributed as dist

from ae.config import AEConfig
from ae.model import LatentAE
from utils.density_kv_integration import reset_density_kv_banks
from utils.temporal_attention_trace import reset_temporal_attention_trace
from utils.retrieval_selection import deterministic_random_indices
from utils.retrieval_profile import summarize_retrieval_profile


def avg_pool(latent_frame: torch.Tensor) -> torch.Tensor:
    """
    Compress a latent frame [B, C, H, W] into a descriptor [B, C] via spatial average pooling.
    
    This is the default compression method for computing latent descriptors
    used in memory token selection. Alternative methods can be swapped in
    by replacing this function.
    
    Args:
        latent_frame: Tensor of shape [B, C, H, W] (single frame latent)
    
    Returns:
        Tensor of shape [B, C] — the compressed descriptor
    """
    return latent_frame.mean(dim=(-2, -1))


def _warp_denoising_schedule(
    denoising_steps: Sequence[int],
    scheduler_timesteps,
    num_train_timestep: int = 1000,
):
    """Map configured training steps onto the scheduler's shifted timeline."""
    steps = tuple(int(step) for step in denoising_steps)
    if not steps:
        raise ValueError("denoising schedule must contain at least one step")
    if any(step < 0 or step > num_train_timestep for step in steps):
        raise ValueError(
            "denoising steps must lie in "
            f"[0, {num_train_timestep}], got {steps}"
        )
    if len(scheduler_timesteps) < num_train_timestep:
        raise ValueError(
            "scheduler timeline is shorter than num_train_timestep: "
            f"{len(scheduler_timesteps)} < {num_train_timestep}"
        )

    warped = []
    for step in steps:
        index = num_train_timestep - step
        if index == len(scheduler_timesteps):
            # The scheduler stores steps [1000, ..., 1]; raw step 0 is the
            # extra terminal value appended by the official implementation.
            warped.append(0.0)
        else:
            warped.append(scheduler_timesteps[index])

    if hasattr(scheduler_timesteps, "new_tensor"):
        return scheduler_timesteps.new_tensor(
            [float(value) for value in warped]
        )
    return tuple(warped)


def _select_denoising_schedule(
    block_index: int,
    denoising_step_list,
    denoising_step_list_first_chunk=None,
):
    """Use the optional first-chunk schedule only for generated block zero."""
    if block_index == 0 and denoising_step_list_first_chunk is not None:
        return denoising_step_list_first_chunk
    return denoising_step_list


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        # Filter pipeline-specific settings out of model_kwargs so they don't reach the
        # WanDiffusionWrapper init.
        model_args_clean = dict(getattr(args, "model_kwargs", {}))
        density_kv_cfg = getattr(args.model_kwargs, "density_kv", None)
        self.memory_selection = str(
            getattr(args.model_kwargs, "memory_selection", "retrieval")
        )
        if self.memory_selection not in {"retrieval", "fixed_prefix", "random"}:
            raise ValueError(
                f"unsupported memory_selection: {self.memory_selection}"
            )
        self.memory_selection_seed = int(
            getattr(args.model_kwargs, "memory_selection_seed", 0)
        )
        self.retrieval_profile_enabled = bool(
            getattr(args, "retrieval_profile", False)
        )
        self.retrieval_profile_warmup_blocks = int(
            getattr(args, "retrieval_profile_warmup_blocks", 0)
        )
        if self.retrieval_profile_warmup_blocks < 0:
            raise ValueError("retrieval_profile_warmup_blocks must be non-negative")
        self.retrieval_profile_log: list[dict[str, float | int | str]] = []
        self.retrieval_profile_summary: dict | None = None
        self.density_kv_enabled = bool(
            getattr(density_kv_cfg, "enabled", False)
            if density_kv_cfg is not None
            else False
        )
        for key in [
            "compression_method",
            "ae_ckpt",
            "recent_exclude",
            "memory_selection",
            "memory_selection_seed",
            "density_kv",
            "temporal_attention_trace",
        ]:
            model_args_clean.pop(key, None)

        self.generator = WanDiffusionWrapper(
            **model_args_clean, is_causal=True) if generator is None else generator
        if self.memory_selection == "fixed_prefix":
            fixed_memory_limit = int(
                getattr(args.model_kwargs, "memory_size", 0)
            )
            for block in getattr(self.generator.model, "blocks", []):
                block.self_attn.fixed_cpu_memory_limit = fixed_memory_limit
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_update_mode = str(
            getattr(args, "denoising_update_mode", "x0_renoise")
        )
        if self.denoising_update_mode not in {"x0_renoise", "flow_euler"}:
            raise ValueError(
                "denoising_update_mode must be 'x0_renoise' or 'flow_euler', "
                f"got {self.denoising_update_mode!r}"
            )
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            self.denoising_step_list = _warp_denoising_schedule(
                args.denoising_step_list,
                self.scheduler.timesteps.cpu(),
            )

        first_chunk_steps = getattr(
            args, "denoising_step_list_first_chunk", None
        )
        if first_chunk_steps is not None:
            self.denoising_step_list_first_chunk = torch.tensor(
                first_chunk_steps, dtype=torch.long
            )
            if args.warp_denoising_step:
                self.denoising_step_list_first_chunk = (
                    _warp_denoising_schedule(
                        first_chunk_steps,
                        self.scheduler.timesteps.cpu(),
                    )
                )
        else:
            self.denoising_step_list_first_chunk = None

        # Default Wan 480p token count. Each inference call replaces this from
        # its actual latent height/width before allocating or indexing caches.
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        # Retrieval autoencoder (optional). compression_method ∈ {"avg_pool", "ae"}.
        self.compression_method = getattr(args.model_kwargs, "compression_method", "avg_pool")
        self.ae_model = None
        if (
            self.compression_method == "ae"
            and not self.density_kv_enabled
            and self.memory_selection not in {"fixed_prefix", "random"}
        ):
            ae_ckpt = getattr(args.model_kwargs, "ae_ckpt", None)
            if ae_ckpt and os.path.exists(ae_ckpt):
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Loading LatentAE from {ae_ckpt} for compression...")
                import dataclasses
                ckpt = torch.load(ae_ckpt, map_location="cpu")
                ae_cfg_dict = ckpt["config"]

                # Sanitize old configs by keeping only fields that exist in the current AEConfig
                valid_keys = {f.name for f in dataclasses.fields(AEConfig)}
                ae_cfg_dict = {k: v for k, v in ae_cfg_dict.items() if k in valid_keys}

                ae_cfg = AEConfig(**ae_cfg_dict)
                self.ae_model = LatentAE(ae_cfg).to(device)
                self.ae_model.load_state_dict(ckpt["model"], strict=False)
                self.ae_model.eval()
            else:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Warning: ae_ckpt {ae_ckpt!r} not found; falling back to avg_pool.")
                self.compression_method = "avg_pool"

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        skip_vae_decode: bool = False,
        text_prompt_schedule: Optional[Sequence[Tuple[int, List[str]]]] = None,
        latent_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        collect_output: bool = True,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block
        if (
            self.retrieval_profile_enabled
            and self.retrieval_profile_warmup_blocks >= num_blocks
        ):
            raise ValueError(
                "retrieval_profile_warmup_blocks must be smaller than num_blocks"
            )
        patch_size = tuple(int(value) for value in self.generator.model.patch_size)
        if len(patch_size) != 3:
            raise ValueError(f"expected a 3D model patch size, got {patch_size}")
        if height % patch_size[1] or width % patch_size[2]:
            raise ValueError(
                "latent spatial shape must be divisible by the model patch size: "
                f"shape=({height}, {width}), patch={patch_size}"
            )
        self.frame_seq_length = (
            height // patch_size[1]
        ) * (width // patch_size[2])
        if return_latents and not collect_output:
            raise ValueError("return_latents requires collect_output=True")
        if not collect_output and not skip_vae_decode:
            raise ValueError("streaming output requires skip_vae_decode=True")

        if self.density_kv_enabled:
            reset_density_kv_banks(self.generator.model)
        reset_temporal_attention_trace(self.generator.model)

        raw_schedule = list(text_prompt_schedule or [(0, text_prompts)])
        if not raw_schedule or int(raw_schedule[0][0]) != 0:
            raise ValueError("text_prompt_schedule must start at latent frame 0")
        conditional_schedule = []
        previous_start = -1
        rng_state_after_first_prompt = None
        for schedule_position, (start_frame, scheduled_prompts) in enumerate(raw_schedule):
            start_frame = int(start_frame)
            if start_frame <= previous_start:
                raise ValueError("text_prompt_schedule starts must be strictly increasing")
            if start_frame % self.num_frame_per_block != 0:
                raise ValueError(
                    "text prompt switches must align to num_frame_per_block"
                )
            if start_frame >= num_output_frames:
                raise ValueError("text prompt switch lies outside the rollout")
            if len(scheduled_prompts) != batch_size:
                raise ValueError("each scheduled prompt list must match the batch size")
            conditional_schedule.append(
                (
                    start_frame,
                    self.text_encoder(text_prompts=list(scheduled_prompts)),
                    list(scheduled_prompts),
                )
            )
            if schedule_position == 0 and len(raw_schedule) > 1:
                rng_state_after_first_prompt = torch.cuda.get_rng_state(noise.device)
            previous_start = start_frame
        if rng_state_after_first_prompt is not None:
            # Pre-encoding future prompts must not perturb the scheduler random
            # stream relative to a one-prompt reference rollout.
            torch.cuda.set_rng_state(rng_state_after_first_prompt, noise.device)
        conditional_dict = conditional_schedule[0][1]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output = None
        if collect_output:
            output_device = torch.device('cpu') if low_memory else noise.device
            output = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=output_device,
                dtype=noise.dtype
            )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        memory_size_cfg = getattr(self.args.model_kwargs, "memory_size", 0)
        kv_policy = ""
        if self.density_kv_enabled and local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local+density-kv, size={local_attn_cfg} frames"
        elif memory_size_cfg > 0 and local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local+cpu_offload, size={local_attn_cfg} frames"
        elif local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        schedule_index = 0
        self.retrieval_profile_log = []
        self.retrieval_profile_summary = None
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # Initialize latent descriptor cache for memory token selection
        self.latent_descriptors = []  # List of [B, C] tensors (or [B, D] for AE)
        self.memory_indices_log = []  # Track memory selection
        # self.compression_method is already set in __init__
        sink_size = getattr(self.args.model_kwargs, "sink_size", 0)
        recent_exclude = getattr(self.args.model_kwargs, "recent_exclude", 0)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        pbar_blocks = tqdm(all_num_frames, desc=f"Generating blocks", disable=(dist.is_initialized() and dist.get_rank() != 0))
        for block_index, current_num_frames in enumerate(pbar_blocks):
            if (
                schedule_index + 1 < len(conditional_schedule)
                and current_start_frame >= conditional_schedule[schedule_index + 1][0]
            ):
                schedule_index += 1
                conditional_dict = conditional_schedule[schedule_index][1]
                # Cross-attention K/V encode the text condition and must be rebuilt.
                # Self-attention KV and Density-KV deliberately remain untouched.
                self._initialize_crossattn_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device,
                )
                scheduled_prompts = conditional_schedule[schedule_index][2]
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(
                        f"[prompt-schedule] latent_frame={current_start_frame} "
                        f"stage={schedule_index} prompt={scheduled_prompts[0]!r}"
                    )
                self.memory_indices_log.append({
                    "query_frame": current_start_frame,
                    "mode": "prompt_switch",
                    "prompt_stage": schedule_index,
                    "prompt": scheduled_prompts[0],
                })
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]

            # Step 2.0: Compute memory_indices from latent descriptors (shared across all layers)
            memory_indices = None
            if memory_size_cfg > 0 and not self.density_kv_enabled:
                # Number of evicted frames in the CPU memory pool (same across all layers)
                num_evicted = len(self.kv_cache1[0].get("cpu_k_frames", []))
                if self.memory_selection == "fixed_prefix":
                    k_sel = min(memory_size_cfg, num_evicted)
                    if k_sel > 0:
                        memory_indices = torch.arange(
                            k_sel,
                            dtype=torch.long,
                            device=noise.device,
                        ).unsqueeze(0).expand(batch_size, -1)
                elif self.memory_selection == "random":
                    num_eligible = max(num_evicted - recent_exclude, 0)
                    if num_eligible > 0:
                        k_sel = min(memory_size_cfg, num_eligible)
                        sampled = deterministic_random_indices(
                            num_eligible=num_eligible,
                            count=k_sel,
                            batch_size=batch_size,
                            seed=self.memory_selection_seed,
                            query_frame=current_start_frame,
                        )
                        memory_indices = torch.tensor(
                            sampled,
                            dtype=torch.long,
                            device=noise.device,
                        )
                        global_frame_indices = memory_indices + sink_size
                        self.memory_indices_log.append({
                            "query_frame": current_start_frame,
                            "mode": "random",
                            "num_evicted": num_evicted,
                            "num_eligible": num_eligible,
                            "selected_pool_indices": memory_indices.cpu().tolist(),
                            "selected_global_frames": global_frame_indices.cpu().tolist(),
                            "selected_similarities": [
                                [0.0] * k_sel for _ in range(batch_size)
                            ],
                            "similarity_available": False,
                            "selection_seed": self.memory_selection_seed,
                        })
                else:
                    # Exclude the `recent_exclude` most-recently-evicted frames from the
                    # candidate pool (they sit right next to the local attention window).
                    num_eligible = max(num_evicted - recent_exclude, 0)
                    if num_eligible > 0 and len(self.latent_descriptors) > 0:
                        k_sel = min(memory_size_cfg, num_eligible)

                        search_start = search_end = None
                        if self.retrieval_profile_enabled:
                            search_start = torch.cuda.Event(enable_timing=True)
                            search_end = torch.cuda.Event(enable_timing=True)
                            search_start.record()

                        evicted_descs = torch.stack([
                            self.latent_descriptors[sink_size + i]
                            for i in range(num_eligible)
                        ], dim=1)  # [B, num_eligible, C]

                        query_desc = self.latent_descriptors[-1].unsqueeze(1)  # [B, 1, C]

                        q_norm = query_desc / (query_desc.norm(dim=-1, keepdim=True) + 1e-8)
                        k_norm = evicted_descs / (evicted_descs.norm(dim=-1, keepdim=True) + 1e-8)
                        sims = torch.bmm(k_norm, q_norm.transpose(1, 2)).squeeze(-1)  # [B, num_eligible]

                        topk_sims, memory_indices = torch.topk(sims, k=k_sel, dim=-1)  # [B, k_sel]

                        if search_end is not None and search_start is not None:
                            search_end.record()
                            search_end.synchronize()
                            self.retrieval_profile_log.append({
                                "kind": "topk_search",
                                "block_index": block_index,
                                "query_frame": current_start_frame,
                                "milliseconds": float(
                                    search_start.elapsed_time(search_end)
                                ),
                            })

                        global_frame_indices = memory_indices + sink_size
                        self.memory_indices_log.append({
                            "query_frame": current_start_frame,
                            "num_evicted": num_evicted,
                            "selected_pool_indices": memory_indices.cpu().tolist(),
                            "selected_global_frames": global_frame_indices.cpu().tolist(),
                            "selected_similarities": topk_sims.cpu().tolist(),
                            "compression_method": self.compression_method,
                        })

            if self.density_kv_enabled:
                first_attn = self.generator.model.blocks[0].self_attn
                first_attn.density_kv_last_stats = None
                first_attn.density_kv_last_processed_count = 0
                for block in self.generator.model.blocks:
                    block.self_attn.density_kv_trace_query_frame = int(
                        current_start_frame
                    )

            current_denoising_list = _select_denoising_schedule(
                block_index,
                self.denoising_step_list,
                self.denoising_step_list_first_chunk,
            )

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(current_denoising_list):
                # print(f"current_timestep: {current_timestep}")

                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(current_denoising_list) - 1:
                    flow_pred, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        memory_indices=memory_indices
                    )
                    next_timestep = current_denoising_list[index + 1]
                    if self.denoising_update_mode == "flow_euler":
                        noisy_input = self.scheduler.step_to(
                            flow_pred.flatten(0, 1),
                            timestep.flatten(0, 1),
                            noisy_input.flatten(0, 1),
                            next_timestep,
                        ).unflatten(0, noisy_input.shape[:2])
                    else:
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    flow_pred, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        memory_indices=memory_indices
                    )
                    if self.denoising_update_mode == "flow_euler":
                        denoised_pred = self.scheduler.step_to(
                            flow_pred.flatten(0, 1),
                            timestep.flatten(0, 1),
                            noisy_input.flatten(0, 1),
                            0,
                        ).unflatten(0, noisy_input.shape[:2])
            # Step 2.2: record the model's output
            if output is not None:
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            if latent_callback is not None:
                latent_callback(current_start_frame, denoised_pred.detach())

            # Step 2.2.1: Store latent descriptors for newly denoised frames
            # denoised_pred shape: [B, current_num_frames, C, H, W]
            if (
                memory_size_cfg > 0
                and not self.density_kv_enabled
                and self.memory_selection not in {"fixed_prefix", "random"}
            ):
                encoding_start = encoding_end = None
                if self.retrieval_profile_enabled:
                    encoding_start = torch.cuda.Event(enable_timing=True)
                    encoding_end = torch.cuda.Event(enable_timing=True)
                    encoding_start.record()
                for f_idx in range(current_num_frames):
                    frame = denoised_pred[:, f_idx]  # [B, C, H, W]
                    if self.compression_method == "ae" and self.ae_model is not None:
                        desc = self.ae_model.encode(frame)  # [B, latent_dim]
                    else:
                        desc = avg_pool(frame)              # [B, C]
                    self.latent_descriptors.append(desc.detach())
                if encoding_end is not None and encoding_start is not None:
                    encoding_end.record()
                    encoding_end.synchronize()
                    self.retrieval_profile_log.append({
                        "kind": "latent_encoding",
                        "block_index": block_index,
                        "query_frame": current_start_frame,
                        "milliseconds": float(
                            encoding_start.elapsed_time(encoding_end)
                        ),
                    })

            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * getattr(self.args, "context_noise", 0.0)
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                memory_indices=memory_indices,
                clean_cache_commit=True,
            )

            if self.memory_selection == "fixed_prefix":
                first_attn = self.generator.model.blocks[0].self_attn
                selected_global_frames = (
                    (memory_indices + sink_size).cpu().tolist()
                    if memory_indices is not None
                    else [[] for _ in range(batch_size)]
                )
                selected_pool_indices = (
                    memory_indices.cpu().tolist()
                    if memory_indices is not None
                    else [[] for _ in range(batch_size)]
                )
                selected_similarities = [
                    [1.0] * len(indices) for indices in selected_pool_indices
                ]
                self.memory_indices_log.append({
                    "query_frame": current_start_frame,
                    "mode": "fixed_prefix",
                    "num_evicted": len(self.kv_cache1[0].get("cpu_k_frames", [])),
                    "selected_pool_indices": selected_pool_indices,
                    "selected_global_frames": selected_global_frames,
                    "selected_similarities": selected_similarities,
                    "attention_memory_tokens": int(
                        getattr(first_attn, "last_memory_tokens", 0)
                    ),
                    "attention_local_tokens": int(
                        getattr(first_attn, "last_local_tokens", 0)
                    ),
                    "attention_total_tokens": int(
                        getattr(first_attn, "last_attention_tokens", 0)
                    ),
                    "compression_method": "fixed_prefix",
                })

            if self.density_kv_enabled:
                first_attn = self.generator.model.blocks[0].self_attn
                density_count = int(getattr(first_attn, "density_kv_active_count", 0))
                density_counts = first_attn.density_kv_bank.counts
                density_count_min = int(density_counts.min().item())
                density_count_max = int(density_counts.max().item())
                density_stats = getattr(first_attn, "density_kv_last_stats", None)
                gate_accepted_mask = (
                    getattr(density_stats, "trace_gate_accepted", None)
                    if density_stats is not None
                    else None
                )
                gate_accepted_counts = (
                    gate_accepted_mask.sum(dim=1, dtype=torch.int32)
                    if gate_accepted_mask is not None
                    else None
                )
                accepted_entries = (
                    int(density_stats.accepted_count[0].item())
                    if density_stats is not None
                    else 0
                )
                candidate_density_mean = (
                    float(density_stats.candidate_density[0].float().mean().item())
                    if density_stats is not None
                    else 0.0
                )
                admission_score_mean = (
                    float(density_stats.energy_delta[0].float().mean().item())
                    if density_stats is not None
                    else 0.0
                )
                decision_group_sizes = (
                    [
                        int(value)
                        for value in density_stats.trace_group_candidate_count[
                            0
                        ].tolist()
                    ]
                    if density_stats is not None
                    and getattr(
                        density_stats, "trace_group_candidate_count", None
                    ) is not None
                    else []
                )
                self.memory_indices_log.append({
                    "query_frame": current_start_frame,
                    "mode": "density_kv",
                    "active_entries_per_head": density_count,
                    "active_entries_per_head_min": density_count_min,
                    "active_entries_per_head_max": density_count_max,
                    "processed_entries": int(
                        getattr(first_attn, "density_kv_last_processed_count", 0)
                    ),
                    "decision_entries": sum(decision_group_sizes),
                    "decision_group_sizes": decision_group_sizes,
                    "accepted_entries": accepted_entries,
                    "accepted_entries_min": (
                        int(density_stats.accepted_count.min().item())
                        if density_stats is not None
                        else 0
                    ),
                    "accepted_entries_max": (
                        int(density_stats.accepted_count.max().item())
                        if density_stats is not None
                        else 0
                    ),
                    "gate_accepted_entries": (
                        int(gate_accepted_counts[0].item())
                        if gate_accepted_counts is not None
                        else accepted_entries
                    ),
                    "gate_accepted_entries_min": (
                        int(gate_accepted_counts.min().item())
                        if gate_accepted_counts is not None
                        else accepted_entries
                    ),
                    "gate_accepted_entries_max": (
                        int(gate_accepted_counts.max().item())
                        if gate_accepted_counts is not None
                        else accepted_entries
                    ),
                    "pruned_after_gate_entries": (
                        int(gate_accepted_counts[0].item()) - accepted_entries
                        if gate_accepted_counts is not None
                        else 0
                    ),
                    "candidate_density_mean": candidate_density_mean,
                    "admission_score_mean": admission_score_mean,
                    "source_tokens_seen": int(
                        getattr(first_attn, "density_kv_source_tokens_seen", 0)
                    ),
                    "attention_memory_tokens": int(
                        getattr(first_attn, "density_kv_last_memory_tokens", 0)
                    ),
                    "attention_local_tokens": int(
                        getattr(first_attn, "density_kv_last_local_tokens", 0)
                    ),
                    "attention_total_tokens": int(
                        getattr(first_attn, "density_kv_last_attention_tokens", 0)
                    ),
                })

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if self.retrieval_profile_enabled:
            self.retrieval_profile_summary = summarize_retrieval_profile(
                self.retrieval_profile_log,
                num_blocks=num_blocks,
                warmup_blocks=self.retrieval_profile_warmup_blocks,
            )
            print(
                "[retrieval-profile] "
                f"{dict((key, value) for key, value in self.retrieval_profile_summary.items() if key != 'raw_samples')}"
            )

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        if skip_vae_decode:
            video = None
            if profile:
                vae_end.record()
                torch.cuda.synchronize()
                vae_time = vae_start.elapsed_time(vae_end)
                total_time = init_time + diffusion_time + vae_time

                print("Profiling results:")
                print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
                print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
                for i, block_time in enumerate(block_times):
                    print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
                print(f"  - VAE decoding skipped")
                print(f"  - Total time: {total_time:.2f} ms")
        else:
            assert output is not None
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=True)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            if profile:
                # End VAE timing and synchronize CUDA
                vae_end.record()
                torch.cuda.synchronize()
                vae_time = vae_start.elapsed_time(vae_end)
                total_time = init_time + diffusion_time + vae_time

                print("Profiling results:")
                print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
                print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
                for i, block_time in enumerate(block_times):
                    print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
                print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
                print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            assert output is not None
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "cpu_k_frames": [],
                "cpu_v_frames": []
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass
