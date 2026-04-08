import torch
from torch import device


# Check for MPS availability (macOS Metal Performance Shaders)
def _has_mps():
    try:
        # For PyTorch >= 1.12
        return torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    except:
        return False


DEVICE: device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if _has_mps() else torch.device("cpu")
)

from .sdk.forecasting import perform_forecasting

__all__ = [
    "DEVICE",
    "perform_forecasting",
]
