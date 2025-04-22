"""MNIST dataloader returning train/val loaders."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloaders(root: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()  # keep pixels in [0,1] for BCE
    train_ds = datasets.MNIST(root, train=True, download=True, transform=transform)
    val_ds   = datasets.MNIST(root, train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )