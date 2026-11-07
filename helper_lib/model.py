"""Model definitions for the helper library (Module 8 class activity).

Diffusion model building blocks are adapted from the Module 8 Diffusion practical,
with these changes:
- 32x32 RGB CIFAR-10 images instead of 64x64 Flowers102 images
- unused layers (BatchNorm in ResidualBlock, embedding_proj in UNet) removed
"""

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Trained weights are stored inside helper_lib/ so the Docker image includes them
WEIGHTS_DIR = Path(__file__).parent / "weights"
DIFFUSION_WEIGHTS_PATH = WEIGHTS_DIR / "diffusion_cifar10.pth"
EBM_WEIGHTS_PATH = WEIGHTS_DIR / "ebm_cifar10.pth"


# ---------------------------------------------------------------------------
# Diffusion model: UNet that predicts the noise in a noisy image
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Turns a noise variance (one number per image) into a vector of sines and cosines."""

    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Frequencies spaced evenly on a log scale between 1 and 1000
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        # x: (B, 1, 1, 1) -> (B, 1, 1, 2 * num_frequencies)
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions plus a skip connection that adds the input back."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 convolution to match channel counts when they differ
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = F.silu(self.conv1(x))  # SiLU is the same as swish: x * sigmoid(x)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Residual blocks, then average pooling to halve the height and width."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)  # saved for the matching UpBlock
        return self.pool(x)


class UpBlock(nn.Module):
    """Upsampling to double the height and width, then residual blocks that use the skips."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)  # concatenate the saved skip connection
            x = block(x)
        return x


class UNet(nn.Module):
    """Predicts the noise in a noisy image, given the noise variance.

    Shapes for 32x32 RGB input (channels x height x width):
    input 3x32x32 -> initial conv 32x32x32 (+32 embedding channels = 64)
    down1 -> 32x16x16, down2 -> 64x8x8, down3 -> 96x4x4 (bottleneck)
    mid -> 128x4x4
    up1 -> 96x8x8, up2 -> 64x16x16, up3 -> 32x32x32, final conv -> 3x32x32
    """

    def __init__(self, image_size=32, num_channels=3, num_frequencies=16):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels

        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies)
        embedding_dim = 2 * num_frequencies  # 32

        self.down1 = DownBlock(32, block_depth=2, in_channels=32 + embedding_dim)
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(96, block_depth=2, in_channels=128)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)  # start by predicting zero noise, as in the class code

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        # Embedding (B, 1, 1, 32) -> (B, 32, 1, 1) -> repeated over every pixel (B, 32, H, W)
        noise_emb = self.embedding(noise_variances).permute(0, 3, 1, 2)
        noise_emb = F.interpolate(noise_emb, size=(self.image_size, self.image_size), mode="nearest")
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Maps diffusion times t in [0, 1] to (noise_rates, signal_rates).

    t = 0 -> almost the clean image (signal 0.95), t = 1 -> almost pure noise (signal 0.02).
    signal_rate**2 + noise_rate**2 = 1, so the noisy image keeps unit variance.
    """
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates, signal_rates


class DiffusionModel(nn.Module):
    """Wraps the UNet with the forward (noising) and reverse (denoising) diffusion processes."""

    def __init__(self, network, schedule_fn, ema_decay=0.999):
        super().__init__()
        self.network = network                      # trained with gradient descent
        self.ema_network = copy.deepcopy(network)   # slow moving average, used for generation
        for p in self.ema_network.parameters():
            p.requires_grad_(False)
        self.ema_decay = ema_decay
        self.schedule_fn = schedule_fn
        # Per-channel normalization statistics, stored as buffers so they are saved with the weights
        self.register_buffer("normalizer_mean", torch.zeros(1, network.num_channels, 1, 1))
        self.register_buffer("normalizer_std", torch.ones(1, network.num_channels, 1, 1))

    def set_normalizer(self, mean, std):
        self.normalizer_mean.copy_(mean.reshape(1, -1, 1, 1))
        self.normalizer_std.copy_(std.reshape(1, -1, 1, 1))

    def normalize(self, x):
        return (x - self.normalizer_mean) / self.normalizer_std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # Main network during training, EMA network for validation and generation
        network = self.network if training else self.ema_network
        pred_noises = network(noisy_images, noise_rates ** 2)
        # Invert x_t = signal * x_0 + noise * eps to estimate the clean image x_0
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def add_noise(self, images):
        # Forward process: pick a random diffusion time per image and mix in Gaussian noise
        noises = torch.randn_like(images)
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        return noisy_images, noises, noise_rates, signal_rates

    def train_step(self, images, optimizer, loss_fn):
        images = self.normalize(images)
        noisy_images, noises, noise_rates, signal_rates = self.add_noise(images)

        # The UNet is trained to predict the noise that was added
        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the EMA copy: ema = decay * ema + (1 - decay) * current weights
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.mul_(self.ema_decay).add_(param, alpha=1.0 - self.ema_decay)
        return loss.item()

    @torch.no_grad()
    def test_step(self, images, loss_fn):
        images = self.normalize(images)
        noisy_images, noises, noise_rates, signal_rates = self.add_noise(images)
        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
        return loss_fn(pred_noises, noises).item()

    @torch.no_grad()
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # Start from pure noise (t = 1) and remove a little noise at each step until t = 0
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        pred_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.full((initial_noise.size(0), 1, 1, 1), 1.0 - step * step_size,
                           device=initial_noise.device)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates,
                                                    training=False)
            # Re-noise the predicted clean image to the next (smaller) noise level
            next_noise_rates, next_signal_rates = self.schedule_fn(t - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    @torch.no_grad()
    def generate(self, num_images, diffusion_steps=20, generator=None):
        device = self.normalizer_mean.device
        size = self.network.image_size
        initial_noise = torch.randn((num_images, self.network.num_channels, size, size),
                                    generator=generator).to(device)
        return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))


# ---------------------------------------------------------------------------
# Energy-based model: CNN that assigns low energy to realistic images
# (adapted from the Module 8 Energy Based Methods practical, for 3-channel CIFAR-10 images)
# ---------------------------------------------------------------------------

def randn_on(shape, device, generator=None):
    """Standard normal noise on `device`. A seeded generator lives on the CPU, so draw there and move."""
    if generator is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, generator=generator).to(device)


class EnergyModel(nn.Module):
    """Maps an image to a single energy value.

    Shapes (channels x height x width):
    3x32x32 -> 16x16x16 -> 32x8x8 -> 64x4x4 -> 64x2x2 -> flatten 256 -> 64 -> 1
    """

    def __init__(self, num_channels=3, image_size=32):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.silu(self.conv1(x))  # SiLU = swish, as in the class code
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = F.silu(self.fc1(x))
        return self.fc2(x)


def langevin_sampling(energy_model, images, steps, step_size, noise_std, generator=None):
    """Langevin dynamics: gradient descent on the PIXELS (not the weights) to lower the energy."""
    was_training = energy_model.training
    energy_model.eval()
    with torch.enable_grad():  # works even if the caller turned gradients off
        for _ in range(steps):
            # Add a little noise; no gradient tracking is needed for this
            with torch.no_grad():
                images = (images + noise_std * randn_on(images.shape, images.device, generator)).clamp(-1.0, 1.0)

            # Track gradients with respect to the input images
            images.requires_grad_(True)
            energy = energy_model(images)
            # d(energy)/d(images); grad_outputs gives each image in the batch its own gradient
            grads, = torch.autograd.grad(energy, images, grad_outputs=torch.ones_like(energy))

            # Step downhill in energy, with gradient clipping for stability
            with torch.no_grad():
                images = (images - step_size * grads.clamp(-0.03, 0.03)).clamp(-1.0, 1.0)
    energy_model.train(was_training)
    return images.detach()


class Buffer:
    """Replay buffer of earlier Langevin samples, so chains continue instead of restarting from noise."""

    def __init__(self, image_shape, batch_size=128, max_size=8192):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.max_size = max_size
        self.examples = torch.rand((batch_size, *image_shape)) * 2 - 1  # uniform noise in [-1, 1]

    def sample_new_exmps(self, energy_model, steps, step_size, noise_std, device):
        self.examples = self.examples.to(device)
        # About 5% of the batch starts from fresh noise, the rest continues from the buffer
        n_new = int((torch.rand(self.batch_size) < 0.05).sum())
        new_rand_imgs = torch.rand((n_new, *self.image_shape), device=device) * 2 - 1
        idx = torch.randint(len(self.examples), (self.batch_size - n_new,), device=device)
        inp_imgs = torch.cat([new_rand_imgs, self.examples[idx]], dim=0)

        new_imgs = langevin_sampling(energy_model, inp_imgs, steps, step_size, noise_std)

        # Newest samples first; keep at most max_size
        self.examples = torch.cat([new_imgs, self.examples], dim=0)[: self.max_size]
        return new_imgs


class EBM(nn.Module):
    """Trains an EnergyModel with contrastive divergence and generates images with Langevin dynamics."""

    def __init__(self, model, alpha=0.1, steps=60, step_size=10.0, noise_std=0.005):
        super().__init__()
        self.model = model
        self.buffer = Buffer(image_shape=(model.num_channels, model.image_size, model.image_size))
        self.alpha = alpha            # weight of the regularization term
        self.steps = steps            # Langevin steps per training batch
        self.step_size = step_size
        self.noise_std = noise_std

    def train_step(self, real_imgs, optimizer):
        # Real images (in [-1, 1]) with a little noise, and fake images from the buffer
        real_imgs = (real_imgs + self.noise_std * torch.randn_like(real_imgs)).clamp(-1.0, 1.0)
        fake_imgs = self.buffer.sample_new_exmps(
            self.model, self.steps, self.step_size, self.noise_std, real_imgs.device)

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0).detach()
        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        # Contrastive divergence: lower the energy of real images, raise it for fake images
        cdiv_loss = real_out.mean() - fake_out.mean()
        # Regularization keeps the energy values close to 0
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()

        return {"loss": loss.item(), "cdiv": cdiv_loss.item(),
                "real": real_out.mean().item(), "fake": fake_out.mean().item()}

    def generate(self, num_images, steps=1024, step_size=10.0, noise_std=0.005, generator=None):
        """Starts from uniform noise and runs a long Langevin chain; returns images in [0, 1].

        Defaults were chosen by comparing samples: 1024 steps with the training noise level (0.005)
        gave clearly smoother images than the class default (256 steps, noise 0.01).
        """
        device = next(self.model.parameters()).device
        shape = (num_images, self.model.num_channels, self.model.image_size, self.model.image_size)
        images = (torch.rand(shape, generator=generator) * 2 - 1).to(device)
        images = langevin_sampling(self.model, images, steps, step_size, noise_std, generator)
        return ((images + 1) / 2).clamp(0.0, 1.0)


def get_model(model_name):
    """Returns a new, untrained model by name (helper library interface from the class activity)."""
    if model_name == "Diffusion":
        unet = UNet(image_size=32, num_channels=3)
        return DiffusionModel(unet, offset_cosine_diffusion_schedule)
    if model_name == "EBM":
        return EBM(EnergyModel(num_channels=3, image_size=32))
    raise ValueError(f"Unknown model name: {model_name}")


# Where each model's trained weights are saved
WEIGHTS_PATHS = {
    "Diffusion": DIFFUSION_WEIGHTS_PATH,
    "EBM": EBM_WEIGHTS_PATH,
}

def load_model(model_name, device="cpu"):
    """Builds a model with get_model() and loads its trained weights, ready for generation."""
    model = get_model(model_name)
    model.load_state_dict(torch.load(WEIGHTS_PATHS[model_name], map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Sanity check: one training step and a short generation on random "images" in [0, 1]
    model = get_model("Diffusion")
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    fake_batch = torch.rand(4, 3, 32, 32)
    print("Training loss:", model.train_step(fake_batch, optimizer, loss_fn))
    samples = model.generate(num_images=4, diffusion_steps=5)
    print("Generated images shape:", samples.shape)
    print("Pixel range:", samples.min().item(), "to", samples.max().item())

    # EBM sanity check: one training step and a short Langevin chain on random images in [-1, 1]
    ebm = get_model("EBM")
    ebm_optimizer = torch.optim.Adam(ebm.model.parameters(), lr=1e-4, betas=(0.0, 0.999))
    real_batch = torch.rand(8, 3, 32, 32) * 2 - 1
    print("EBM metrics:", ebm.train_step(real_batch, ebm_optimizer))
    ebm_samples = ebm.generate(num_images=4, steps=10)
    print("EBM samples shape:", ebm_samples.shape)
    print("EBM pixel range:", ebm_samples.min().item(), "to", ebm_samples.max().item())