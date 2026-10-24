from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.cnn_model import WEIGHTS_PATH, ImageClassifier
from app.rnn_model import RNN_VOCAB_PATH, RNN_WEIGHTS_PATH, TextGenerator
from app.gan_model import GAN_WEIGHTS_PATH, DigitGenerator

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()
# Load the trained CNN once at startup (None if train_cnn.py has not been run yet)
image_classifier = ImageClassifier() if WEIGHTS_PATH.exists() else None
# Load the trained LSTM once at startup (None if train_rnn.py has not been run yet)
text_generator = (
    TextGenerator() if RNN_WEIGHTS_PATH.exists() and RNN_VOCAB_PATH.exists() else None
)
# Load the trained GAN generator once at startup (None if train_gan.py has not been run yet)
digit_generator = DigitGenerator() if GAN_WEIGHTS_PATH.exists() else None


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class RNNTextGenerationRequest(BaseModel):
    start_word: str = Field(..., examples=["the count of monte cristo"])
    length: int = Field(30, ge=1, le=200)
    temperature: float = Field(1.0, gt=0, le=5)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.get("/embedding")
def get_embedding(word: str):
    try:
        embedding = embedding_model.get_embedding(word)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"word": word, "dimension": len(embedding), "embedding": embedding}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Upload an image (jpg/png/...) and get its predicted CIFAR-10 class."""
    if image_classifier is None:
        raise HTTPException(status_code=503, detail="Model weights not found. Run train_cnn.py first.")
    image_bytes = await file.read()
    try:
        result = image_classifier.predict(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")
    return {"filename": file.filename, **result}


@app.post("/generate_with_rnn")
def generate_text_with_rnn(request: RNNTextGenerationRequest):
    """Generate text with the LSTM model trained on The Count of Monte Cristo."""
    if text_generator is None:
        raise HTTPException(status_code=503, detail="LSTM model not found. Run train_rnn.py first.")
    try:
        generated_text = text_generator.generate(
            request.start_word, request.length, request.temperature
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"generated_text": generated_text}


@app.get(
    "/generate_digits",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def generate_digits(
    num_images: int = Query(16, ge=1, le=64),
    seed: int | None = Query(None, ge=0),
):
    """Generate hand-written digits with the GAN trained on MNIST (returns a PNG grid)."""
    if digit_generator is None:
        raise HTTPException(status_code=503, detail="GAN model not found. Run train_gan.py first.")
    png_bytes = digit_generator.generate_png(num_images=num_images, seed=seed)
    return Response(content=png_bytes, media_type="image/png")