"""Image generation for the helper library (Module 8 class activity)."""

import io
import math

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def images_to_png(images):
    """Arranges a batch of images with values in [0, 1] into a square-ish grid, as PNG bytes."""
    nrow = math.ceil(math.sqrt(images.size(0)))
    grid = make_grid(images.cpu(), nrow=nrow, padding=2)
    buffer = io.BytesIO()
    to_pil_image(grid).save(buffer, format="PNG")
    return buffer.getvalue()


@torch.no_grad()
def generate_samples(model, device, num_samples=10, diffusion_steps=100, seed=None):
    """Generates num_samples images with a trained DiffusionModel and returns a PNG grid.

    Starts from standard normal noise and runs the reverse diffusion for diffusion_steps steps.
    A fixed seed gives the same images every time; no seed gives new images on each call.
    """
    model.to(device)
    model.eval()
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    images = model.generate(num_samples, diffusion_steps=diffusion_steps, generator=rng)
    return images_to_png(images)


@torch.no_grad()
def generate_ebm_samples(model, device, num_samples=10, steps=1024, seed=None):
    """Generates num_samples images with a trained EBM and returns a PNG grid.

    Starts from uniform noise and runs Langevin dynamics (gradient descent on the pixels)
    for `steps` steps. A fixed seed gives the same images every time.
    """
    model.to(device)
    model.eval()
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    images = model.generate(num_samples, steps=steps, generator=rng)
    return images_to_png(images)