"""Validation step used by training & standalone runs."""
import torch


def validate(model, loader, device, loss_fn, beta):
    model.eval()
    total = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            total += loss_fn(recon, x, mu, logvar, beta).item()
    return total / len(loader.dataset)