"""Train the diffusion model on CIFAR-10 and save the weights.

Usage (from the repo root):
    uv run python train_diffusion.py
    uv run python train_diffusion.py --epochs 1
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from helper_lib.model import DIFFUSION_WEIGHTS_PATH, get_model
from helper_lib.trainer import train_diffusion

SAMPLES_DIR = Path("data") / "diffusion_samples"  # sample grids for checking progress (ignored by git)

# Per-channel mean and std of the CIFAR-10 training set
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616])


def get_device():
    # Use the Apple GPU (MPS) or an NVIDIA GPU if available, otherwise CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. CIFAR-10 with pixels in [0, 1]; the model normalizes them itself
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 2. Model, loss and optimizer (same choices as the class practical)
    model = get_model("Diffusion")
    model.set_normalizer(CIFAR10_MEAN, CIFAR10_STD)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=args.lr, weight_decay=1e-4)

    # After each epoch, save the same 64 generated images so progress is easy to compare
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    def save_samples(epoch):
        model.eval()
        samples = model.generate(64, diffusion_steps=20, generator=torch.Generator().manual_seed(0))
        save_image(samples, SAMPLES_DIR / f"epoch_{epoch:02d}.png", nrow=8)

    # 3. Train
    train_diffusion(model, train_loader, criterion, optimizer, device=device,
                    epochs=args.epochs, val_loader=test_loader, on_epoch_end=save_samples)

    # 4. Save all weights (network, EMA network, normalizer) on CPU
    DIFFUSION_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, DIFFUSION_WEIGHTS_PATH)
    print(f"Saved weights to {DIFFUSION_WEIGHTS_PATH}")
    print(f"Sample grids saved in {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()