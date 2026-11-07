"""Train the energy-based model on CIFAR-10 and save the weights.

Usage (from the repo root):
    uv run python train_ebm.py
    uv run python train_ebm.py --epochs 1
"""

import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from helper_lib.model import EBM_WEIGHTS_PATH, get_model
from helper_lib.trainer import train_ebm

SAMPLES_DIR = Path("data") / "ebm_samples"  # sample grids for checking progress (ignored by git)


def get_device():
    # Use the Apple GPU (MPS) or an NVIDIA GPU if available, otherwise CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. CIFAR-10 scaled to [-1, 1] (the energy model and Langevin sampler assume this range)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # 2. Model and optimizer (same choices as the class practical)
    model = get_model("EBM")
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr, betas=(0.0, 0.999))

    # After each epoch, save the newest buffer samples and images generated from fresh noise
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    def save_samples(epoch):
        buffer_imgs = (model.buffer.examples[:64] + 1) / 2
        save_image(buffer_imgs.clamp(0, 1), SAMPLES_DIR / f"epoch_{epoch:02d}_buffer.png", nrow=8)
        generated = model.generate(64, steps=256, generator=torch.Generator().manual_seed(0))
        save_image(generated, SAMPLES_DIR / f"epoch_{epoch:02d}_generated.png", nrow=8)

    # 3. Train
    train_ebm(model, train_loader, optimizer, device=device, epochs=args.epochs,
              on_epoch_end=save_samples)

    # 4. Save the energy network weights on CPU (the replay buffer is not part of the model)
    EBM_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, EBM_WEIGHTS_PATH)
    print(f"Saved weights to {EBM_WEIGHTS_PATH}")
    print(f"Sample grids saved in {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()