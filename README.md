# sps_genai – Assignments 1 & 2

FastAPI server with three models:
- **Bigram text generator** (Module 3 class activity)
- **Word embeddings** using spaCy `en_core_web_md` (300-dim vectors)
- **CNN image classifier** trained on CIFAR-10 (Assignment 2)

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

The trained CNN weights (`app/weights/cnn_cifar10.pth`) are committed to the repo,
so the Docker image works without retraining.

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/` | – | Hello World |
| POST | `/generate` | `{"start_word": "the", "length": 10}` | generated text |
| GET | `/embedding` | query param `word`, e.g. `?word=apple` | `word`, `dimension`, `embedding` (list of 300 floats) |
| POST | `/classify` | image file upload (form field `file`, jpg/png/...) | `predicted_class`, `confidence`, `top_k` |

### Examples

```bash
curl "http://127.0.0.1:8000/embedding?word=apple"
curl -X POST -F "file=@cat.jpeg" http://127.0.0.1:8000/classify
```

Example `/classify` response:

```json
{
  "filename": "cat.jpeg",
  "predicted_class": "cat",
  "confidence": 0.5467,
  "top_k": [
    {"class": "cat", "probability": 0.5467},
    {"class": "truck", "probability": 0.4473},
    {"class": "airplane", "probability": 0.0059}
  ]
}
```

- `/embedding` returns **400** if more than one word is given, **404** if the word has no vector
- `/classify` returns **400** if the file is not a valid image, **503** if the model weights are missing

## Assignment 2: CNN

Architecture (`app/cnn_model.py`), as specified in the assignment:

| Layer | Output shape |
|---|---|
| Input (RGB) | 3 × 64 × 64 |
| Conv2d(16, 3×3, stride 1, padding 1) + ReLU | 16 × 64 × 64 |
| MaxPool2d(2×2, stride 2) | 16 × 32 × 32 |
| Conv2d(32, 3×3, stride 1, padding 1) + ReLU | 32 × 32 × 32 |
| MaxPool2d(2×2, stride 2) | 32 × 16 × 16 |
| Flatten | 8192 |
| Linear(100) + ReLU | 100 |
| Linear(10) | 10 |

CIFAR-10 images (32×32) are resized to 64×64 and normalized with the CIFAR-10
channel mean/std. The same preprocessing is used for training and inference.

### Retrain the model (optional)

```bash
uv sync
uv run python -m app.cnn_model   # sanity check: prints the architecture and output shape
uv run python train_cnn.py       # downloads CIFAR-10, trains 10 epochs, saves the weights
```

Result after 10 epochs (Adam, lr 1e-3, batch size 64): train accuracy 92.1%,
test accuracy 64.9%. The gap shows some overfitting, which is expected since the
specified architecture has no dropout or batch normalization.

## Project structure

```
app/
├── main.py             # FastAPI app and endpoints
├── bigram_model.py     # BigramModel class
├── embedding_model.py  # EmbeddingModel class (spaCy)
├── cnn_model.py        # CNN architecture, preprocessing, ImageClassifier
└── weights/
    └── cnn_cifar10.pth # trained CNN weights
train_cnn.py            # trains the CNN on CIFAR-10
Dockerfile
pyproject.toml / uv.lock
```