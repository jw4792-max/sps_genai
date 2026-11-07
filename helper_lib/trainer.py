"""Training loops for the helper library (Module 8 class activity)."""

from tqdm import tqdm


def train_diffusion(model, data_loader, criterion, optimizer, device="cpu", epochs=10,
                    val_loader=None, on_epoch_end=None):
    """Runs several epochs of diffusion training and returns the trained model.

    model:        a DiffusionModel from get_model("Diffusion")
    criterion:    loss between predicted noise and real noise (e.g. nn.L1Loss())
    val_loader:   optional, used to report a validation loss after each epoch
    on_epoch_end: optional function called with the epoch number (e.g. to save sample images)
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100)
        for batch_number, (images, _) in enumerate(progress):  # labels are not used
            images = images.to(device)
            total_loss += model.train_step(images, optimizer, criterion)
            progress.set_postfix(loss=f"{total_loss / (batch_number + 1):.4f}")

        message = f"Epoch {epoch + 1}: train loss {total_loss / len(data_loader):.4f}"
        if val_loader is not None:
            model.eval()
            val_losses = [model.test_step(images.to(device), criterion) for images, _ in val_loader]
            message += f", val loss {sum(val_losses) / len(val_losses):.4f}"
        print(message)

        if on_epoch_end is not None:
            on_epoch_end(epoch + 1)
    return model


def train_ebm(model, data_loader, optimizer, device="cpu", epochs=10, on_epoch_end=None):
    """Runs several epochs of energy-based model training and returns the trained model.

    model:        an EBM from get_model("EBM")
    data_loader:  real images scaled to [-1, 1]
    on_epoch_end: optional function called with the epoch number (e.g. to save sample images)

    There is no separate criterion: the contrastive divergence loss is computed inside
    model.train_step, because it needs fake images produced by Langevin sampling.
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        totals = {"loss": 0.0, "cdiv": 0.0, "real": 0.0, "fake": 0.0}
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100)
        for batch_number, (images, _) in enumerate(progress):  # labels are not used
            metrics = model.train_step(images.to(device), optimizer)
            for key in totals:
                totals[key] += metrics[key]
            progress.set_postfix(cdiv=f"{totals['cdiv'] / (batch_number + 1):.4f}")

        averages = {key: value / len(data_loader) for key, value in totals.items()}
        print(f"Epoch {epoch + 1}: " + ", ".join(f"{k} {v:.4f}" for k, v in averages.items()))

        if on_epoch_end is not None:
            on_epoch_end(epoch + 1)
    return model