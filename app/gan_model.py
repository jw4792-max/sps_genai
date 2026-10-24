import io
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

Z_DIM = 100  # size of the input noise vector

# Trained generator weights are saved inside app/ so the Docker image includes them
GAN_WEIGHTS_PATH = Path(__file__).parent / "weights" / "gan_generator.pth"


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        # Fully connected layer: noise vector -> 7 * 7 * 128 values
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        # Upsampling part: 7x7 -> 14x14 -> 28x28
        # Comments show the output shape of each layer as (channels, height, width)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 1 x 28 x 28
            nn.Tanh(),  # pixel values in [-1, 1], matching the normalized MNIST images
        )

    def forward(self, z):
        x = self.fc(z)                  # (batch, 100) -> (batch, 6272)
        x = x.view(-1, 128, 7, 7)       # reshape -> (batch, 128, 7, 7)
        return self.deconv(x)           # -> (batch, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Downsampling part: 28x28 -> 14x14 -> 7x7
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 64 x 14 x 14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        # Single output score (logit); the sigmoid that turns it into a
        # real/fake probability is applied inside BCEWithLogitsLoss during training
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 128 * 7 * 7 = 6272
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))  # -> (batch, 1)


class DigitGenerator:
    """Loads the trained generator once and turns random noise into a PNG grid of digits."""

    def __init__(self, weights_path=GAN_WEIGHTS_PATH):
        self.model = Generator()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()  # inference mode: BatchNorm uses the statistics learned in training

    @torch.no_grad()
    def generate_png(self, num_images=16, seed=None):
        # A fixed seed gives the same digits every time; no seed gives new digits on each call
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        z = torch.randn(num_images, Z_DIM, generator=rng)
        images = self.model(z)            # (num_images, 1, 28, 28), values in [-1, 1]
        images = (images + 1) / 2         # rescale to [0, 1] for saving as an image
        nrow = math.ceil(math.sqrt(num_images))  # arrange the digits in a square-ish grid
        grid = make_grid(images, nrow=nrow, padding=2)
        buffer = io.BytesIO()
        to_pil_image(grid).save(buffer, format="PNG")
        return buffer.getvalue()          # raw PNG bytes, ready to send from the API


if __name__ == "__main__":
    # Sanity check: 4 noise vectors -> 4 fake images -> 4 real/fake scores
    gen = Generator()
    disc = Discriminator()
    z = torch.randn(4, Z_DIM)
    fake_images = gen(z)
    scores = disc(fake_images)
    print(gen)
    print(disc)
    print("Generated images shape:", fake_images.shape)
    print("Discriminator output shape:", scores.shape)
    print("Pixel range:", fake_images.min().item(), "to", fake_images.max().item())