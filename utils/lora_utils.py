# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import torch
import peft
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig
)


def patch_peft_tensor_parallel_compat():
    """Support the older PEFT build used by the LongLive runtime."""
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


def configure_lora_for_model(transformer, model_name, lora_config, is_main_process=True):
    """Configure LoRA for a WanDiffusionWrapper model
    
    Args:
        transformer: The transformer model to apply LoRA to
        model_name: 'generator' or 'fake_score'
        lora_config: LoRA configuration
        is_main_process: Whether this is the main process (for logging)
    
    Returns:
        lora_model: The LoRA-wrapped model
    """
    # Find all Linear modules in WanAttentionBlock modules
    target_linear_modules = set()
    
    # Define the specific modules we want to apply LoRA to
    if model_name == 'generator':
        adapter_target_modules = ['CausalWanAttentionBlock']
    elif model_name == 'fake_score':
        adapter_target_modules = ['WanAttentionBlock']
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)
    
    target_linear_modules = list(target_linear_modules)
    
    if is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        if getattr(lora_config, 'verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")
    
    # Create LoRA config
    adapter_type = lora_config.get('type', 'lora')
    if adapter_type == 'lora':
        peft_config = peft.LoraConfig(
            r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', None) or lora_config.get('rank', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
    
    # Apply LoRA to the transformer
    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()
    
    return lora_model


def gather_lora_state_dict(lora_model):
    with FSDP.state_dict_type(
        lora_model,                  
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
    ):
        full = lora_model.state_dict()
    return get_peft_model_state_dict(lora_model, state_dict=full)


def load_lora_checkpoint(lora_model, lora_state_dict, model_name, is_main_process=True):
    """Load LoRA weights from state dict
    
    Args:
        lora_model: The LoRA-wrapped model
        lora_state_dict: LoRA state dict to load
        model_name: 'generator' or 'critic'
        is_main_process: Whether this is the main process (for logging)
    """
    if is_main_process:
        print(f"Loading LoRA {model_name} weights: {len(lora_state_dict)} keys in checkpoint")
    
    peft.set_peft_model_state_dict(lora_model, lora_state_dict)
    
    if is_main_process:
        print(f"LoRA {model_name} weights loaded successfully")


def configure_and_load_lora_for_model(
    transformer,
    *,
    model_name,
    lora_config,
    checkpoint_path,
    is_main_process=True,
):
    """Attach PEFT with the configured shape and load a LongLive LoRA state."""
    patch_peft_tensor_parallel_compat()
    lora_model = configure_lora_for_model(
        transformer,
        model_name=model_name,
        lora_config=lora_config,
        is_main_process=is_main_process,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = (
        checkpoint["generator_lora"]
        if isinstance(checkpoint, dict) and "generator_lora" in checkpoint
        else checkpoint
    )
    load_lora_checkpoint(
        lora_model,
        state,
        model_name=model_name,
        is_main_process=is_main_process,
    )
    return lora_model


def merge_lora_into_base_model(lora_model, *, is_main_process=True):
    """Merge a loaded PEFT adapter into dense weights and remove PEFT wrappers."""
    if not isinstance(lora_model, peft.PeftModel):
        raise TypeError(
            "merge_lora_into_base_model expected a peft.PeftModel, "
            f"got {type(lora_model).__name__}"
        )
    merged_model = lora_model.merge_and_unload(safe_merge=True)
    remaining = [name for name, _ in merged_model.named_parameters() if "lora_" in name]
    if remaining:
        raise RuntimeError(f"LoRA merge left adapter parameters behind: {remaining[:8]}")
    if is_main_process:
        parameter_count = sum(parameter.numel() for parameter in merged_model.parameters())
        print(f"LoRA merged into dense base weights: {parameter_count:,} parameters")
    return merged_model
