import io
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# CIFAR-10 class names, in label order (label 0 = airplane, ..., label 9 = truck)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMAGE_SIZE = 64
# Trained weights are saved inside app/ so the Docker image includes them
WEIGHTS_PATH = Path(__file__).parent / "weights" / "cnn_cifar10.pth"

# Per-channel mean and std of the CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Shared preprocessing for both training and inference:
# CIFAR-10 images are 32x32, but the architecture expects 64x64 input
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional part: extracts image features
        # Comments show the output shape of each layer as (channels, height, width)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),   # 16 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 16 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                   # 32 x 16 x 16
        )
        # Fully connected part: classifies the extracted features
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 32 * 16 * 16 = 8192
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ImageClassifier:
    """Loads the trained CNN once and classifies raw image bytes."""

    def __init__(self, weights_path=WEIGHTS_PATH):
        self.model = CNN(num_classes=len(CLASSES))
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()  # switch to inference mode

    @torch.no_grad()
    def predict(self, image_bytes, top_k=3):
        # Decode the uploaded file and force 3 channels (handles PNG with alpha, grayscale, etc.)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Apply the same preprocessing as in training, then add a batch dimension
        x = preprocess(image).unsqueeze(0)  # shape: 1 x 3 x 64 x 64
        # Turn the 10 raw scores (logits) into probabilities that sum to 1
        probs = torch.softmax(self.model(x), dim=1)[0]
        top_probs, top_idx = probs.topk(top_k)
        return {
            "predicted_class": CLASSES[top_idx[0].item()],
            "confidence": round(top_probs[0].item(), 4),
            "top_k": [
                {"class": CLASSES[i.item()], "probability": round(p.item(), 4)}
                for p, i in zip(top_probs, top_idx)
            ],
        }

if __name__ == "__main__":
    # Sanity check: pass 4 random 64x64 RGB "images" through the model
    model = CNN()
    fake_images = torch.randn(4, 3, 64, 64)
    output = model(fake_images)
    print(model)
    print("Output shape:", output.shape)