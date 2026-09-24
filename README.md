# sps_genai – Assignment 1

FastAPI server with two models:
- **Bigram text generator** (Module 3 class activity)
- **Word embeddings** using spaCy `en_core_web_md` (300-dim vectors)

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/` | – | Hello World |
| POST | `/generate` | `{"start_word": "the", "length": 10}` | generated text |
| GET | `/embedding` | query param `word`, e.g. `?word=apple` | `word`, `dimension`, `embedding` (list of 300 floats) |

### Example

```bash
curl "http://127.0.0.1:8000/embedding?word=apple"
```

- Returns **400** if more than one word is given
- Returns **404** if the word has no vector in the model

## Project structure

```
app/
├── main.py             # FastAPI app and endpoints
├── bigram_model.py     # BigramModel class
└── embedding_model.py  # EmbeddingModel class (spaCy)
```