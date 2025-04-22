import torch
from torchvision.utils import make_grid


def recon_grid(original: torch.Tensor, reconstructed: torch.Tensor):
    original, reconstructed = original.cpu(), reconstructed.cpu()
    grid = torch.cat([original[:8], reconstructed[:8]], 0)  # 8 originals + 8 recons
    return make_grid(grid, nrow=8, pad_value=1.0)