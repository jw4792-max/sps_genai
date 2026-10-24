"""Train the GAN on MNIST and save the generator weights.

Usage (from the repo root):
    uv run python train_gan.py
    uv run python train_gan.py --epochs 1
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from app.gan_model import GAN_WEIGHTS_PATH, Z_DIM, Discriminator, Generator

SAMPLES_DIR = Path("data") / "gan_samples"  # sample grids for checking progress (ignored by git)


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
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load MNIST and scale pixels to [-1, 1] to match the generator's Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Models, loss function and optimizers
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    # BCEWithLogitsLoss = sigmoid (score -> probability) + binary cross-entropy
    criterion = nn.BCEWithLogitsLoss()
    # Adam with beta1 = 0.5 is the standard choice for training GANs (DCGAN paper)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Fixed noise: the same 64 vectors every epoch, so the sample grids show progress
    fixed_noise = torch.randn(64, Z_DIM, device=device)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Training loop
    for epoch in range(args.epochs):
        gen.train()
        disc.train()
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=100)
        for real, _ in progress:  # labels are not needed: the GAN only learns what digits look like
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)   # 1 = real
            fake_labels = torch.zeros(batch_size, 1, device=device)  # 0 = fake

            # --- Train the discriminator: real images -> 1, generated images -> 0 ---
            noise = torch.randn(batch_size, Z_DIM, device=device)
            fake = gen(noise)
            loss_disc_real = criterion(disc(real), real_labels)
            # detach(): do not update the generator in this step
            loss_disc_fake = criterion(disc(fake.detach()), fake_labels)
            loss_disc = loss_disc_real + loss_disc_fake

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # --- Train the generator: make the discriminator output 1 for generated images ---
            loss_gen = criterion(disc(fake), real_labels)

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            progress.set_postfix(D_loss=f"{loss_disc.item():.3f}", G_loss=f"{loss_gen.item():.3f}")

        # Save a grid of 64 generated digits after each epoch
        gen.eval()
        with torch.no_grad():
            samples = gen(fixed_noise)
        save_image(samples, SAMPLES_DIR / f"epoch_{epoch + 1:02d}.png", nrow=8, normalize=True)

    # 4. Save the generator weights (on CPU). The API only needs the generator.
    GAN_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in gen.state_dict().items()}, GAN_WEIGHTS_PATH)
    print(f"Saved generator weights to {GAN_WEIGHTS_PATH}")
    print(f"Sample grids saved in {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()