import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_md"):
        self.nlp = spacy.load(model_name)

    def get_embedding(self, word):
        doc = self.nlp(word.strip())
        if len(doc) != 1:
            raise ValueError("Please provide a single word.")
        token = doc[0]
        if not token.has_vector:
            raise KeyError(f"No embedding found for '{word}'.")
        return token.vector.tolist()