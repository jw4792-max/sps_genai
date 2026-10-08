from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.cnn_model import WEIGHTS_PATH, ImageClassifier

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


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


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