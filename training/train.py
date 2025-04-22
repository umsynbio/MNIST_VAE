import yaml
import os
import wandb
import torch
from torch import nn, optim
from dataloaders.mnist_dataloader import get_mnist_dataloaders
from models.vae import VAE
from training.validate import validate
from utils.visualization import recon_grid


def elbo_loss(recon_x, x, mu, logvar, beta):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def train(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    wandb.init(
        project=cfg["wandb"]["project"], config=cfg
    )

    device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else "cpu")
    train_loader, val_loader = get_mnist_dataloaders(
        cfg["data"]["root"], cfg["training"]["batch_size"]
    )

    model = VAE(
        latent_dim=cfg["model"]["latent_dim"], hidden_dims=cfg["model"]["hidden_dims"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        running_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = elbo_loss(recon, data, mu, logvar, cfg["training"]["beta"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader.dataset)  # type: ignore
        val_loss = validate(
            model, val_loader, device, elbo_loss, cfg["training"]["beta"]
        )

        # Log scalars & sample reconstructions
        sample, _ = next(iter(val_loader))
        sample = sample.to(device)[:8]
        model.eval()
        with torch.no_grad():
            sample_recon, _, _ = model(sample)
        grid = recon_grid(sample, sample_recon)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "reconstructions": wandb.Image(grid, caption=f"Epoch {epoch}"),
            }
        )

        print(f"Epoch {epoch}: train {train_loss:.4f}, val {val_loss:.4f}")

    # Save & log final model
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/vae_mnist.pt"
    torch.save(model.state_dict(), ckpt_path)
    artifact = wandb.Artifact("mnist-vae-model", type="model")
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    train(p.parse_args().config)
