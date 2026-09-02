from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def wrap_path_string(path_string: str) -> Path:
    return Path(path_string)


def get_attn_implementation(pretrained_model_name_or_path: str) -> str:
    if "gemma-4" in pretrained_model_name_or_path:
        # flash_attention_2 causes RuntimeError: FlashAttention only supports head dimensions up to 256
        return "sdpa"
    else:
        return "flash_attention_2"


def get_effective_batch_size(per_device_train_batch_size: int, gradient_accumulation_steps: int) -> int:
    if dist.is_initialized():
        world_size = dist.get_world_size()
    elif torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 1
    return per_device_train_batch_size * world_size * gradient_accumulation_steps


def set_omegaconf_resolver() -> None:
    # replace=True: overwrite the same name resolver
    OmegaConf.register_new_resolver("wrap_path_string", wrap_path_string, replace=True)
    OmegaConf.register_new_resolver("get_attn_implementation", get_attn_implementation, replace=True)
    OmegaConf.register_new_resolver("get_effective_batch_size", get_effective_batch_size, replace=True)
    OmegaConf.register_new_resolver("get_device_count", torch.cuda.device_count, replace=True)
