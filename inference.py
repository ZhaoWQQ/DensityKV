# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import torch
import os
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset
from utils.misc import set_seed

from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.streaming_rollout import LatentShardWriter, decode_shards_to_mp4

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single GPU mode on device {device}")

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)


# Initialize pipeline. The released LongLive checkpoints use the few-step
# x0/re-noise path; pre-distillation ODE checkpoints use causal diffusion.
inference_pipeline = str(
    getattr(config, "inference_pipeline", "causal_few_step")
)
if inference_pipeline == "causal_few_step":
    pipeline = CausalInferencePipeline(config, device=device)
elif inference_pipeline == "causal_diffusion":
    pipeline = CausalDiffusionInferencePipeline(config, device=device)
else:
    raise ValueError(
        "inference_pipeline must be 'causal_few_step' or "
        f"'causal_diffusion', got {inference_pipeline!r}"
    )

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        strict_checkpoint_load = bool(
            getattr(config, "strict_checkpoint_load", False)
        )
        if strict_checkpoint_load:
            pipeline.generator.load_state_dict(cleaned_state_dict, strict=True)
            if local_rank == 0:
                print("Generator checkpoint loaded with strict key/shape validation.")
        else:
            missing, unexpected = pipeline.generator.load_state_dict(
                cleaned_state_dict, strict=False
            )
            if local_rank == 0:
                if len(missing) > 0:
                    print(
                        f"[Warning] {len(missing)} parameters are missing when loading "
                        f"checkpoint: {missing[:8]} ..."
                    )
                if len(unexpected) > 0:
                    print(
                        f"[Warning] {len(unexpected)} unexpected parameters encountered "
                        f"when loading checkpoint: {unexpected[:8]} ..."
                    )
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import (
    configure_lora_for_model,
    merge_lora_into_base_model,
)
import peft
from utils.density_kv_integration import (
    attach_density_kv_banks,
    density_kv_config_enabled,
    export_density_kv_lineage,
)
from utils.temporal_attention_trace import (
    attach_temporal_attention_trace,
    export_temporal_attention_trace,
    temporal_attention_trace_config_enabled,
)


def _patch_peft_tensor_parallel_compat():
    """Skip PEFT TP sharding on environments without the optional TP classes."""
    try:
        from transformers.integrations.tensor_parallel import EmbeddingParallel  # noqa: F401
        return
    except (ImportError, AttributeError):
        pass

    import peft.utils.save_and_load as peft_save_and_load

    def _no_tp_shard(model, state_dict, adapter_name):
        return None

    peft_save_and_load._maybe_shard_state_dict_for_tp = _no_tp_shard
    peft.set_peft_model_state_dict.__globals__["_maybe_shard_state_dict_for_tp"] = _no_tp_shard


_patch_peft_tensor_parallel_compat()

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # 在加载基础权重后，对 generator 的 transformer 模型应用 LoRA 包装
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # 加载 LoRA 权重（如果提供了 lora_ckpt）
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # 兼容包含 `generator_lora` 键或直接是 LoRA state dict 两种格式
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

    if bool(getattr(config, "merge_lora_into_base", False)):
        pipeline.generator.model = merge_lora_into_base_model(
            pipeline.generator.model,
            is_main_process=(local_rank == 0),
        )


# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)
pipeline.generator.eval()

density_kv_cfg = getattr(config.model_kwargs, "density_kv", None)
if density_kv_config_enabled(density_kv_cfg):
    attached = attach_density_kv_banks(
        pipeline.generator.model,
        density_kv_cfg,
        batch_size=int(config.num_samples),
    )
    if local_rank == 0:
        capacity = int(getattr(density_kv_cfg, "capacity", 8192))
        process_all = bool(
            getattr(density_kv_cfg, "process_all_candidates", False)
        )
        update_mode = (
            f"{str(getattr(density_kv_cfg, 'full_update_mode', 'frozen_snapshot'))}, "
            f"all candidates in chunks of "
            f"{int(getattr(density_kv_cfg, 'update_chunk_size', 512))}"
            if process_all
            else "bounded admission"
        )
        capacity_label = (
            f"initial_capacity={capacity} entries/head"
            if process_all
            and str(
                getattr(density_kv_cfg, "full_update_mode", "frozen_snapshot")
            ) == "append_only_density"
            else f"capacity={capacity} entries/head"
        )
        local_window_frames = int(
            getattr(density_kv_cfg, "local_window_frames", -1)
        )
        local_policy = (
            f"fixed {local_window_frames} frames"
            if local_window_frames >= 0
            else "shared attention budget"
        )
        source_token_limit = int(
            getattr(density_kv_cfg, "source_token_limit", -1)
        )
        logical_precommit = bool(
            getattr(
                density_kv_cfg,
                "logical_precommit",
                not bool(getattr(density_kv_cfg, "verify_fixed_prefix", False)),
            )
        )
        source_policy = (
            f"freeze after {source_token_limit} source tokens"
            if source_token_limit >= 0
            else "continuous updates"
        )
        group_count_reduce = str(
            getattr(density_kv_cfg, "append_group_count_reduce", "min")
        )
        legacy_chunk_grouping = str(
            getattr(density_kv_cfg, "legacy_chunk_grouping", "contiguous")
        )
        temporal_rope_mode = str(
            getattr(density_kv_cfg, "temporal_rope_mode", "zero")
        )
        distance_metric = str(
            getattr(density_kv_cfg, "distance_metric", "squared_l2")
        )
        query_response_rank = int(
            getattr(density_kv_cfg, "query_response_rank", 64)
        )
        legacy_growth_gate = bool(
            getattr(density_kv_cfg, "legacy_density_growth_gate", False)
        )
        growth_limit = float(
            getattr(density_kv_cfg, "append_density_growth_limit", 2.0)
        )
        normalized_groups = int(
            getattr(density_kv_cfg, "legacy_normalized_group_count", -1)
        )
        cleanup_divisor = int(
            getattr(density_kv_cfg, "legacy_cleanup_divisor", -1)
        )
        print(
            f"[density-kv] attached {attached} per-layer banks; "
            f"{capacity_label}; mode={update_mode}; "
            f"grouping={legacy_chunk_grouping}; "
            f"growth_gate={legacy_growth_gate}@{growth_limit:g}; "
            f"normalized_groups={normalized_groups}; "
            f"cleanup_divisor={cleanup_divisor}; "
            f"temporal_rope={temporal_rope_mode}; "
            f"distance={distance_metric}; query_rank={query_response_rank}; "
            f"local={local_policy}; "
            f"logical_precommit={logical_precommit}; "
            f"source={source_policy}; "
            f"group_count_reduce={group_count_reduce}"
        )

latent_height = int(getattr(config, "latent_height", 60))
latent_width = int(getattr(config, "latent_width", 104))
if latent_height <= 0 or latent_width <= 0:
    raise ValueError("latent_height and latent_width must be positive")
patch_size = tuple(int(value) for value in pipeline.generator.model.patch_size)
if len(patch_size) != 3:
    raise ValueError(f"expected a 3D model patch size, got {patch_size}")
if latent_height % patch_size[1] or latent_width % patch_size[2]:
    raise ValueError(
        "latent spatial shape must be divisible by the model patch size: "
        f"shape=({latent_height}, {latent_width}), patch={patch_size}"
    )
spatial_tokens_per_frame = (
    latent_height // patch_size[1]
) * (latent_width // patch_size[2])
temporal_attention_trace_cfg = getattr(
    config.model_kwargs,
    "temporal_attention_trace",
    None,
)
if temporal_attention_trace_config_enabled(temporal_attention_trace_cfg):
    attached = attach_temporal_attention_trace(
        pipeline.generator.model,
        temporal_attention_trace_cfg,
        batch_size=int(config.num_samples),
        frame_seq_length=spatial_tokens_per_frame,
        max_frames=int(config.num_output_frames),
    )
    if local_rank == 0:
        denoise_calls = list(
            getattr(temporal_attention_trace_cfg, "denoise_calls", [3])
        )
        print(
            "[temporal-attention-trace] "
            f"attached_layers={attached}; frames={int(config.num_output_frames)}; "
            f"tokens/frame={spatial_tokens_per_frame}; "
            f"denoise_calls={denoise_calls}"
        )
density_candidates_per_update = (
    int(config.num_frame_per_block) * spatial_tokens_per_frame
)
if density_kv_config_enabled(density_kv_cfg):
    update_chunk_size = int(getattr(density_kv_cfg, "update_chunk_size", 512))
    density_update_tail = density_candidates_per_update % update_chunk_size
    expected_candidates = int(
        getattr(density_kv_cfg, "expected_candidates_per_update", -1)
    )
    if expected_candidates >= 0 and expected_candidates != density_candidates_per_update:
        raise ValueError(
            "density candidate count mismatch: "
            f"expected={expected_candidates}, actual={density_candidates_per_update}"
        )
    if bool(
        getattr(density_kv_cfg, "require_chunk_aligned_candidates", False)
    ) and density_update_tail:
        raise ValueError(
            "density candidates are not chunk aligned: "
            f"candidates={density_candidates_per_update}, "
            f"chunk={update_chunk_size}, tail={density_update_tail}"
        )
    if local_rank == 0:
        repeat_tail = int(
            getattr(density_kv_cfg, "legacy_repeat_tail_when_full", 0)
        )
        normalized_group_count = int(
            getattr(density_kv_cfg, "legacy_normalized_group_count", -1)
        )
        cleanup_divisor = int(
            getattr(density_kv_cfg, "legacy_cleanup_divisor", -1)
        )
        cleanup_alignment = int(
            getattr(density_kv_cfg, "legacy_cleanup_alignment", 8)
        )
        normalized_schedule = ""
        if normalized_group_count > 0:
            cleanup_size = 0
            if cleanup_divisor > 0:
                cleanup_size = max(
                    cleanup_alignment,
                    int(
                        round(
                            density_candidates_per_update
                            / cleanup_divisor
                            / cleanup_alignment
                        )
                        * cleanup_alignment
                    ),
                )
            coarse_count = density_candidates_per_update - cleanup_size
            base_size, larger_groups = divmod(
                coarse_count, normalized_group_count
            )
            normalized_sizes = [
                base_size + (index < larger_groups)
                for index in range(normalized_group_count)
            ]
            if cleanup_size:
                normalized_sizes.append(cleanup_size)
            normalized_schedule = f"; normalized_schedule={normalized_sizes}"
        print(
            "[spatial-shape] "
            f"latent={latent_height}x{latent_width}; "
            f"output={latent_height * 8}x{latent_width * 8}; "
            f"tokens/frame={spatial_tokens_per_frame}; "
            f"density-candidates/update={density_candidates_per_update}; "
            f"groups={density_candidates_per_update // update_chunk_size}; "
            f"tail={density_update_tail}; repeat-tail-when-full={repeat_tail}"
            f"{normalized_schedule}"
        )

extended_prompt_path = config.data_path
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


idx_offset = int(getattr(config, "idx_offset", 0))

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    if bool(getattr(config, "reset_seed_per_prompt", False)):
        # Keep prompt-to-prompt comparisons paired when one loaded model serves
        # several prompts in sequence. This is opt-in so historical configs
        # preserve their original continuous RNG stream.
        set_seed(config.seed + local_rank)
        if local_rank == 0:
            print(
                f"[prompt-seed-reset] dataset_index={i} "
                f"seed={config.seed + local_rank}"
            )
    idx = batch_data['idx'].item() + idx_offset

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts'][0]

    # Check if we should skip existing files
    if getattr(config, 'skip_existing', False):
        # Determine model type for filename consistency
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
        
        all_samples_exist = True
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4')
            if not os.path.exists(output_path):
                all_samples_exist = False
                break
        if all_samples_exist:
            continue

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    scheduler_noise_reference_frames = int(
        getattr(config, "scheduler_noise_reference_frames", 0)
    )
    if scheduler_noise_reference_frames < 0:
        raise ValueError("scheduler_noise_reference_frames must be non-negative")
    rng_state_before_initial_noise = (
        torch.cuda.get_rng_state(device)
        if scheduler_noise_reference_frames > 0
        else None
    )
    sampled_noise = torch.randn(
        [
            config.num_samples,
            config.num_output_frames,
            16,
            latent_height,
            latent_width,
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    if scheduler_noise_reference_frames > 0:
        # A longer initial-noise allocation has the same prefix but advances the
        # global CUDA RNG farther, changing every later scheduler randn_like.
        # Rewind and consume the reference-length allocation so a long rollout
        # continues the exact random stream of the reference experiment.
        assert rng_state_before_initial_noise is not None
        torch.cuda.set_rng_state(rng_state_before_initial_noise, device)
        reference_noise = torch.randn(
            [
                config.num_samples,
                scheduler_noise_reference_frames,
                16,
                latent_height,
                latent_width,
            ],
            device=device,
            dtype=torch.bfloat16,
        )
        del reference_noise
        print(
            "[rng-replay] scheduler RNG advanced with "
            f"{scheduler_noise_reference_frames} reference latent frames"
        )

    print("sampled_noise.device", sampled_noise.device)
    print("prompts", prompts)

    streaming_cfg = getattr(config, "streaming_rollout", None)
    streaming_enabled = bool(
        getattr(streaming_cfg, "enabled", False)
        if streaming_cfg is not None
        else False
    )
    if streaming_enabled:
        if config.num_samples != 1:
            raise ValueError("streaming rollout currently requires num_samples=1")
        schedule_items = list(getattr(streaming_cfg, "prompt_schedule", []))
        if not schedule_items:
            raise ValueError("streaming_rollout.prompt_schedule cannot be empty")
        prompt_schedule = []
        schedule_manifest = []
        for stage_index, item in enumerate(schedule_items):
            start_frame = int(item.start_latent_frame)
            stage_prompt = str(item.prompt)
            prompt_schedule.append((start_frame, [stage_prompt]))
            schedule_manifest.append({
                "stage": stage_index,
                "start_latent_frame": start_frame,
                "start_seconds": float(start_frame) / 4.0,
                "prompt": stage_prompt,
            })

        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
        stem = f"rank{rank}-{idx}-0_{model_type}"
        artifact_dir = Path(config.output_folder) / f"{stem}_stream"
        latent_dir = artifact_dir / "latents"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "prompt_schedule.json").write_text(
            json.dumps(schedule_manifest, indent=2), encoding="utf-8"
        )
        shard_writer = LatentShardWriter(
            latent_dir,
            shard_frames=int(getattr(streaming_cfg, "latent_shard_frames", 120)),
        )
        generation_ok = False
        try:
            pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=False,
                low_memory=low_memory,
                profile=False,
                skip_vae_decode=True,
                text_prompt_schedule=prompt_schedule,
                latent_callback=shard_writer,
                collect_output=False,
            )
            generation_ok = True
        finally:
            shard_paths = shard_writer.close()
        if not generation_ok:
            raise RuntimeError("streaming rollout generation did not complete")

        memory_log_path = artifact_dir / "memory_log.json"
        memory_log_path.write_text(
            json.dumps(pipeline.memory_indices_log, indent=2), encoding="utf-8"
        )
        del sampled_noise
        torch.cuda.empty_cache()

        output_path = Path(config.output_folder) / f"{stem}.mp4"
        decode_stats = decode_shards_to_mp4(
            vae=pipeline.vae,
            shard_paths=shard_paths,
            output_path=output_path,
            device=device,
            dtype=torch.bfloat16,
            chunk_frames=int(getattr(streaming_cfg, "vae_chunk_frames", 12)),
            fps=int(getattr(streaming_cfg, "fps", 16)),
            crf=int(getattr(streaming_cfg, "crf", 18)),
        )
        summary = {
            "output": str(output_path),
            "num_latent_frames": int(config.num_output_frames),
            "prompt_schedule": schedule_manifest,
            **decode_stats,
        }
        (artifact_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"[streaming-rollout] completed: {summary}")
        pipeline.vae.model.clear_cache()
        continue


    retrieval_profile_only = bool(
        getattr(config, 'retrieval_profile_only', False)
    )
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
        skip_vae_decode=retrieval_profile_only,
    )
    if retrieval_profile_only:
        if getattr(pipeline, 'retrieval_profile_summary', None) is None:
            raise RuntimeError(
                'retrieval_profile_only requires a completed retrieval profile'
            )
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = 'lora'
        elif getattr(config, 'use_ema', False):
            model_type = 'ema'
        else:
            model_type = 'regular'
        profile_path = os.path.join(
            config.output_folder,
            f'rank{rank}-{idx}-0_{model_type}_retrieval_profile.json',
        )
        with open(profile_path, 'w', encoding='utf-8') as handle:
            json.dump(pipeline.retrieval_profile_summary, handle, indent=2)
        if local_rank == 0:
            print(f'Saved retrieval-only profile to {profile_path}')
        del latents, sampled_noise
        pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()
        continue
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts + idx_offset:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)

            # Save memory selection log
            if hasattr(pipeline, 'memory_indices_log') and pipeline.memory_indices_log:
                import json
                log_path = output_path.replace('.mp4', '_memory_log.json')
                with open(log_path, 'w') as f:
                    json.dump(pipeline.memory_indices_log, f, indent=2)
                if local_rank == 0:
                    print(f"Saved memory selection log to {log_path}")

                lineage_payload = export_density_kv_lineage(
                    pipeline.generator.model
                )
                if lineage_payload is not None:
                    lineage_path = output_path.replace(
                        '.mp4',
                        '_density_lineage.pt',
                    )
                    torch.save(lineage_payload, lineage_path)
                    if local_rank == 0:
                        print(f"Saved density KV lineage to {lineage_path}")

                # Save memory selection visualization
                try:
                    viz_path = output_path.replace('.mp4', '_memory_viz.png')
                    density_entries = [
                        entry for entry in pipeline.memory_indices_log
                        if entry.get('mode') == 'density_kv'
                    ]
                    if density_entries:
                        query_frames = [entry['query_frame'] for entry in density_entries]
                        active_entries = [
                            entry['active_entries_per_head'] for entry in density_entries
                        ]
                        accepted_entries = [
                            entry['accepted_entries'] for entry in density_entries
                        ]
                        processed_entries = [
                            entry.get('processed_entries', 0) for entry in density_entries
                        ]
                        attention_local_tokens = [
                            entry.get('attention_local_tokens', 0)
                            for entry in density_entries
                        ]
                        attention_total_tokens = [
                            entry.get('attention_total_tokens', 0)
                            for entry in density_entries
                        ]
                        plt.figure(figsize=(10, 6))
                        plt.plot(
                            query_frames,
                            active_entries,
                            marker='o',
                            label='Active KV entries / head',
                        )
                        plt.bar(
                            query_frames,
                            accepted_entries,
                            alpha=0.35,
                            width=max(1, getattr(config, 'num_frame_per_block', 3) * 0.6),
                            label='Accepted this block',
                        )
                        plt.plot(
                            query_frames,
                            processed_entries,
                            linestyle='--',
                            label='Processed this block',
                        )
                        plt.plot(
                            query_frames,
                            attention_local_tokens,
                            linestyle=':',
                            label='Local attention tokens',
                        )
                        plt.plot(
                            query_frames,
                            attention_total_tokens,
                            linewidth=2,
                            label='Total attention tokens',
                        )
                        plt.xlabel('Query Frame Index')
                        plt.ylabel('KV Entries')
                        plt.title('Density-Limited KV Bank Occupancy')
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(viz_path)
                        plt.close()
                    else:
                        query_frames = []
                        mem_frames = []
                        sims = []
                        for entry in pipeline.memory_indices_log:
                            q = entry['query_frame']
                            mf = entry['selected_global_frames'][0]
                            s = entry['selected_similarities'][0]
                            for m, sim in zip(mf, s):
                                query_frames.append(q)
                                mem_frames.append(m)
                                sims.append(sim)

                    if not density_entries and query_frames:
                        plt.figure(figsize=(10, 6))
                        sc = plt.scatter(query_frames, mem_frames, c=sims, cmap='viridis', s=30, alpha=0.7)
                        plt.colorbar(sc, label='Cosine Similarity')
                        # Draw causality reference: query frame index
                        plt.plot([0, max(query_frames)], [0, max(query_frames)], 'r--', alpha=0.3, label='Current Frame')
                        plt.xlabel('Query Frame Index')
                        plt.ylabel('Memory Frame Index (Global)')
                        plt.title(f'Memory Selection Visualization (Method: {pipeline.compression_method})')
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(viz_path)
                        plt.close()
                    if os.path.exists(viz_path) and local_rank == 0:
                        print(f"Saved memory selection visualization to {viz_path}")
                except Exception as e:
                    if local_rank == 0:
                        print(f"Failed to create memory visualization: {e}")

            if getattr(pipeline, 'retrieval_profile_summary', None) is not None:
                profile_path = output_path.replace('.mp4', '_retrieval_profile.json')
                with open(profile_path, 'w', encoding='utf-8') as handle:
                    json.dump(pipeline.retrieval_profile_summary, handle, indent=2)
                if local_rank == 0:
                    print(f"Saved retrieval profile to {profile_path}")

            temporal_trace_payload = export_temporal_attention_trace(
                pipeline.generator.model
            )
            if temporal_trace_payload is not None:
                temporal_trace_path = output_path.replace(
                    '.mp4',
                    '_temporal_attention.pt',
                )
                torch.save(temporal_trace_payload, temporal_trace_path)
                if local_rank == 0:
                    print(
                        "Saved temporal attention trace to "
                        f"{temporal_trace_path}"
                    )

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
if dist.is_initialized():
    dist.destroy_process_group()
