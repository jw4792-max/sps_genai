import json
from pathlib import Path

import torch
import torch.nn as nn

# Trained files are saved inside app/ so the Docker image includes them
WEIGHTS_DIR = Path(__file__).parent / "weights"
RNN_WEIGHTS_PATH = WEIGHTS_DIR / "lstm_text.pth"
RNN_VOCAB_PATH = WEIGHTS_DIR / "lstm_vocab.json"


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        # Embedding: maps each word index to a dense vector of size embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM: reads the sequence word by word and keeps a hidden state ("memory")
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Fully connected: turns each hidden state into one score per vocabulary word
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)             # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        x, hidden = self.lstm(x, hidden)  # -> (batch, seq_len, hidden_dim)
        x = self.fc(x)                    # -> (batch, seq_len, vocab_size)
        return x, hidden


class TextGenerator:
    """Loads the trained LSTM and vocabulary once and generates text from a seed."""

    def __init__(self, weights_path=RNN_WEIGHTS_PATH, vocab_path=RNN_VOCAB_PATH):
        self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        self.model = LSTMModel(vocab_size=len(self.vocab))
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()  # switch to inference mode

    @torch.no_grad()
    def generate(self, seed_text, length=30, temperature=1.0):
        words = seed_text.lower().split()
        if not words:
            raise ValueError("Seed text must contain at least one word.")
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")

        # Words not seen during training are mapped to <UNK>
        input_ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # shape: 1 x seq_len
        hidden = None

        for _ in range(length):
            output, hidden = self.model(x, hidden)
            # Scores for the word after the last position, scaled by temperature
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            words.append(self.inv_vocab.get(next_id, "<UNK>"))
            # The hidden state already remembers the earlier words,
            # so only the newly generated word is fed in next
            x = torch.tensor([[next_id]], dtype=torch.long)

        return " ".join(words)


if __name__ == "__main__":
    # Sanity check: a fake batch of 4 sequences, each 30 word indices long
    model = LSTMModel(vocab_size=1000)
    fake_batch = torch.randint(0, 1000, (4, 30))
    output, (h, c) = model(fake_batch)
    print(model)
    print("Output shape:", output.shape)
    print("Hidden state shape:", h.shape)