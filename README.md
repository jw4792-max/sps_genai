# sps_genai – Assignments 1–4 and Module 6 / Module 8 Class Activities

FastAPI server with seven models:
- **Bigram text generator** (Module 3 class activity)
- **Word embeddings** using spaCy `en_core_web_md` (300-dim vectors)
- **CNN image classifier** trained on CIFAR-10 (Assignment 2)
- **LSTM text generator** trained on *The Count of Monte Cristo* (Module 6 class activity)
- **GAN hand-written digit generator** trained on MNIST (Assignment 3)
- **Diffusion model** (UNet) trained on CIFAR-10 (Assignment 4)
- **Energy-based model** with Langevin sampling, trained on CIFAR-10 (Assignment 4)

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

The trained weights in `app/weights/` are committed to the repo,
so the Docker image works without retraining.

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/` | – | Hello World |
| POST | `/generate` | `{"start_word": "the", "length": 10}` | text from the bigram model |
| GET | `/embedding` | query param `word`, e.g. `?word=apple` | `word`, `dimension`, `embedding` (list of 300 floats) |
| POST | `/classify` | image file upload (form field `file`, jpg/png/...) | `predicted_class`, `confidence`, `top_k` |
| POST | `/generate_with_rnn` | `{"start_word": "the ship", "length": 30, "temperature": 0.7}` | text from the LSTM model |
| GET | `/generate_digits` | query params `num_images` (1–64, default 16), `seed` (optional) | PNG image: grid of generated digits |
| GET | `/generate_diffusion` | query params `num_images` (1–64, default 16), `diffusion_steps` (1–500, default 100), `seed` (optional) | PNG image: grid of generated CIFAR-10-like images |
| GET | `/generate_ebm` | query params `num_images` (1–64, default 16), `steps` (1–2048, default 1024), `seed` (optional) | PNG image: grid of generated CIFAR-10-like images |

### Examples

```bash
curl "http://127.0.0.1:8000/embedding?word=apple"
curl -X POST -F "file=@cat.jpeg" http://127.0.0.1:8000/classify
curl -X POST http://127.0.0.1:8000/generate_with_rnn \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the ship", "length": 30, "temperature": 0.7}'
curl "http://127.0.0.1:8000/generate_digits?num_images=16&seed=42" -o digits.png
```

`/generate_digits` can also be opened directly in a browser:
http://127.0.0.1:8000/generate_digits?num_images=16&seed=42
(the same `seed` always gives the same digits; leave it out for new digits on every request).

- `/embedding` returns **400** if more than one word is given, **404** if the word has no vector
- `/classify` returns **400** if the file is not a valid image, **503** if the model weights are missing
- `/generate_with_rnn` returns **422** if `length` is outside 1–200 or `temperature` is outside (0, 5],
  **400** if `start_word` is empty, **503** if the model files are missing
- `/generate_digits` returns **422** if `num_images` is outside 1–64 or `seed` is negative,
  **503** if the generator weights are missing
- `/generate_diffusion` and `/generate_ebm` return **422** if a parameter is out of range,
  **503** if the model weights are missing


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

```bash
uv run python -m app.cnn_model   # sanity check: prints the architecture and output shape
uv run python train_cnn.py       # downloads CIFAR-10, trains 10 epochs, saves the weights
```

Result after 10 epochs (Adam, lr 1e-3, batch size 64): train accuracy 92.1%,
test accuracy 64.9%. The gap shows some overfitting, which is expected since the
specified architecture has no dropout or batch normalization.

## Module 6 Class Activity: LSTM text generator

Replaces the bigram model with a recurrent neural network (`app/rnn_model.py`):
Embedding(vocab, 100) → LSTM(100 → 128) → Linear(128 → vocab).

- Training data: Chapters 1–4 of *The Count of Monte Cristo* (Project Gutenberg),
  punctuation removed and lowercased
- Each training example is a 30-word sequence; the target is the same sequence shifted by one word
- Generation feeds the seed words in, then samples one word at a time from the softmax
  distribution, passing the LSTM hidden state forward. Lower `temperature` gives more
  conservative text, higher gives more random text.

```bash
uv run python -m app.rnn_model   # sanity check: prints the architecture and output shapes
uv run python train_rnn.py       # downloads the book, trains 15 epochs, saves weights + vocabulary
```

## Assignment 3: GAN

Architecture (`app/gan_model.py`), as specified in the assignment:

**Generator**

| Layer | Output shape |
|---|---|
| Input: noise vector | 100 |
| Linear(100 → 7·7·128), reshape | 128 × 7 × 7 |
| ConvTranspose2d(128 → 64, k=4, s=2, p=1) + BatchNorm2d + ReLU | 64 × 14 × 14 |
| ConvTranspose2d(64 → 1, k=4, s=2, p=1) + Tanh | 1 × 28 × 28 |

**Discriminator**

| Layer | Output shape |
|---|---|
| Input: image | 1 × 28 × 28 |
| Conv2d(1 → 64, k=4, s=2, p=1) + LeakyReLU(0.2) | 64 × 14 × 14 |
| Conv2d(64 → 128, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2) | 128 × 7 × 7 |
| Flatten + Linear(6272 → 1) | 1 (real/fake score) |

Training (`train_gan.py`):
- MNIST pixels are scaled to [-1, 1] to match the generator's Tanh output
- Standard GAN loss: `BCEWithLogitsLoss`, which applies the sigmoid to the discriminator's
  score to get the real/fake probability (numerically more stable than Sigmoid + BCELoss)
- Each batch: the discriminator is trained to output 1 for real images and 0 for generated
  images, then the generator is trained to make the discriminator output 1 for generated images
- Adam (lr 2e-4, betas (0.5, 0.999)), batch size 128, 20 epochs
- Only the generator weights are saved, since the API only needs the generator

```bash
uv run python -m app.gan_model   # sanity check: prints both networks and output shapes
uv run python train_gan.py       # downloads MNIST, trains 20 epochs, saves the generator weights
```

## Assignment 4: Diffusion and energy-based models (helper library)

Following the Module 6 / Module 8 class activities, the new models live in a reusable
helper library, and the API only calls into it:

- `helper_lib/model.py`: network definitions and `get_model("Diffusion")` / `get_model("EBM")`,
  plus `load_model(name)` to load the trained weights
- `helper_lib/trainer.py`: `train_diffusion(...)` and `train_ebm(...)`
- `helper_lib/generator.py`: `generate_samples(...)` (diffusion) and `generate_ebm_samples(...)`

**Diffusion model** (adapted from the Module 8 Diffusion practical, for 32×32 RGB CIFAR-10):
- UNet with residual Down/Up blocks and skip connections: 32×32 → 16×16 → 8×8 → 4×4 bottleneck → 32×32;
  the noise variance is encoded with a sinusoidal embedding and concatenated to the input
- Offset cosine diffusion schedule; the UNet predicts the added noise (L1 loss)
- AdamW (lr 1e-3, weight decay 1e-4), batch size 128, 20 epochs; EMA weights (decay 0.999) are used for generation
- Final train / validation loss: 0.162 / 0.162

**Energy-based model** (adapted from the Module 8 Energy Based Methods practical, 3 input channels):
- CNN with SiLU (swish) activations mapping a 3×32×32 image to a single energy value
- Trained with contrastive divergence (energy of real images minus energy of Langevin samples)
  plus a small regularization term, using a replay buffer of earlier samples
- Langevin sampling performs gradient descent on the **input pixels** (`requires_grad_(True)` on the images,
  `torch.autograd.grad` with respect to them), not on the network weights
- Adam (lr 1e-4, betas (0.0, 0.999)), batch size 128, 20 epochs
- Generation runs 1024 Langevin steps from uniform noise (noise std 0.005); these defaults were chosen by
  comparing samples, and gave clearly smoother images than 256 steps

```bash
uv run python -m helper_lib.model   # sanity check: one training step + short generation for both models
uv run python train_diffusion.py    # trains 20 epochs (~30 min on Apple MPS), saves helper_lib/weights/diffusion_cifar10.pth
uv run python train_ebm.py --epochs 20   # trains 20 epochs (~15 min on Apple MPS), saves helper_lib/weights/ebm_cifar10.pth
```

## Project structure

```
app/
├── main.py             # FastAPI app and endpoints
├── bigram_model.py     # BigramModel class
├── embedding_model.py  # EmbeddingModel class (spaCy)
├── cnn_model.py        # CNN architecture, preprocessing, ImageClassifier
├── rnn_model.py        # LSTM architecture, TextGenerator
├── gan_model.py        # Generator, Discriminator, DigitGenerator
└── weights/            # CNN, LSTM and GAN weights
helper_lib/
├── model.py            # UNet, DiffusionModel, EnergyModel, EBM, get_model(), load_model()
├── trainer.py          # train_diffusion(), train_ebm()
├── generator.py        # generate_samples(), generate_ebm_samples()
└── weights/
    ├── diffusion_cifar10.pth  # trained diffusion model (network, EMA network, normalizer)
    └── ebm_cifar10.pth        # trained energy model
train_cnn.py            # trains the CNN on CIFAR-10
train_rnn.py            # trains the LSTM on The Count of Monte Cristo
train_gan.py            # trains the GAN on MNIST
train_diffusion.py      # trains the diffusion model on CIFAR-10
train_ebm.py            # trains the energy-based model on CIFAR-10
Dockerfile
pyproject.toml / uv.lock
```