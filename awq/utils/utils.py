import gc
import importlib
import torch
import accelerate


ipex_available = importlib.util.find_spec("intel_extension_for_pytorch") is not None
try:
    import triton as tl
    triton_available = True
except ImportError:
    triton_available = False


def clear_memory(weight=None):
    if weight is not None:
        del weight
    # gc.collect()
    # torch.cuda.empty_cache()


def compute_memory_used_pct(device):
    memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)
    memory_pct = (
        memory_used
        / (torch.cuda.get_device_properties(device).total_memory / (1024**3))
        * 100
    )
    return memory_pct


def get_best_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    elif torch.xpu.is_available():
        return "xpu:0"
    else:
        return "cpu"