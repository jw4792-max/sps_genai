"""Train the LSTM text generator on The Count of Monte Cristo and save the weights.

Usage (from the repo root):
    uv run python train_rnn.py
    uv run python train_rnn.py --epochs 1
"""

import argparse
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app.rnn_model import RNN_VOCAB_PATH, RNN_WEIGHTS_PATH, LSTMModel

TEXT_URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
TEXT_CACHE = Path("data") / "monte_cristo.txt"
SEQ_LEN = 30
MAX_VOCAB = 10000


def get_device():
    # Use the Apple GPU (MPS) or an NVIDIA GPU if available, otherwise CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_text():
    # Download the book once and cache it in ./data (ignored by git)
    if not TEXT_CACHE.exists():
        TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(TEXT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response:
            TEXT_CACHE.write_text(response.read().decode("utf-8"), encoding="utf-8")
    return TEXT_CACHE.read_text(encoding="utf-8")


def tokenize(text):
    # Same preprocessing as the class practical: keep the first chapters,
    # remove punctuation, lowercase, split on whitespace
    # Skip the table of contents: the second "Chapter 1." is where the story starts
    start_idx = text.find("Chapter 1.", text.find("Chapter 1.") + 1)
    end_idx = text.rfind("Chapter 5.")
    if start_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx]
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().split()


def build_vocab(tokens):
    # Index 0 and 1 are reserved for padding and unknown words;
    # the other indices are assigned by word frequency (most common first)
    counter = Counter(tokens)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(MAX_VOCAB - 2))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


class TextDataset(Dataset):
    # Each example is a 30-word input sequence and the same sequence shifted by one word
    def __init__(self, encoded):
        self.encoded = encoded

    def __len__(self):
        return len(self.encoded) - SEQ_LEN

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx:idx + SEQ_LEN], dtype=torch.long)
        y = torch.tensor(self.encoded[idx + 1:idx + SEQ_LEN + 1], dtype=torch.long)
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Text -> tokens -> vocabulary -> integer ids
    tokens = tokenize(load_text())
    vocab = build_vocab(tokens)
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    print(f"Tokens: {len(tokens)}, vocabulary size: {len(vocab)}")

    loader = DataLoader(TextDataset(encoded), batch_size=args.batch_size, shuffle=True)

    # 2. Model, loss function and optimizer
    model = LSTMModel(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=100)
        for batch_number, (inputs, targets) in enumerate(progress):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            # Flatten (batch, seq_len, vocab) -> (batch * seq_len, vocab) for the loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(avg_loss=f"{total_loss / (batch_number + 1):.4f}")

    # 4. Save the weights (on CPU) and the vocabulary used during training
    RNN_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, RNN_WEIGHTS_PATH)
    RNN_VOCAB_PATH.write_text(json.dumps(vocab), encoding="utf-8")
    print(f"Saved weights to {RNN_WEIGHTS_PATH}")
    print(f"Saved vocabulary to {RNN_VOCAB_PATH}")


if __name__ == "__main__":
    main()