import random
import re
from collections import defaultdict, Counter


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        text = " ".join(corpus)
        self.vocab, self.bigram_probs = self._analyze_bigrams(text, frequency_threshold)

    @staticmethod
    def _tokenize(text, frequency_threshold=None):
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        counts = Counter(tokens)
        return [t for t in tokens if counts[t] >= frequency_threshold]

    def _analyze_bigrams(self, text, frequency_threshold=None):
        words = self._tokenize(text, frequency_threshold)
        bigram_counts = Counter(zip(words[:-1], words[1:]))
        unigram_counts = Counter(words)
        bigram_probs = defaultdict(dict)
        for (w1, w2), count in bigram_counts.items():
            bigram_probs[w1][w2] = count / unigram_counts[w1]
        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, length=10):
        current = start_word.lower()
        words = [current]
        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current)
            if not next_words:
                break
            current = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            words.append(current)
        return " ".join(words)