import importlib.util
import random
from os import environ

import numpy as np
import torch


# Adapted from HuggingFace Transformers (Apache License 2.0)
# cf. https://github.com/huggingface/transformers/blob/v4.50.3/src/transformers/trainer_utils.py#L60-L125
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    environ["CUDA_LAUNCH_BLOCKING"] = "1"
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # The environment variable required to enable deterministic mode on Ascend NPUs.
    environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    environ["HCCL_DETERMINISTIC"] = "1"

    environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    if vllm_is_available() is False:
        torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def vllm_is_available() -> bool:
    return importlib.util.find_spec("vllm") is not None
