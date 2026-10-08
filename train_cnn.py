"""Train the CNN on CIFAR-10 and save the weights.

Usage (from the repo root):
    uv run python train_cnn.py
    uv run python train_cnn.py --epochs 1
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.cnn_model import CNN, WEIGHTS_PATH, preprocess


def get_device():
    # Use the Apple GPU (MPS) or an NVIDIA GPU if available, otherwise CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device):
    # Compute accuracy on a dataset without updating the weights
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load CIFAR-10 (downloaded to ./data the first time)
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=preprocess
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=preprocess
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 2. Model, loss function and optimizer
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=100)
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()              # reset gradients from the previous step
            outputs = model(images)            # forward pass
            loss = criterion(outputs, labels)  # compare predictions with true labels
            loss.backward()                    # backpropagation
            optimizer.step()                   # update the weights

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{correct / total:.3f}")

        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}: train acc {correct / total:.3f}, test acc {test_acc:.3f}")

    # 4. Save the weights (moved to CPU so they can be loaded anywhere, e.g. in Docker)
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()