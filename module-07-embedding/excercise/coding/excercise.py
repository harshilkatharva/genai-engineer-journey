import logging
import math
from collections.abc import Callable
from typing import Any

import nltk  # type: ignore[import-untyped]

# 1


def cosine_similarity(X: list[float], Y: list[float]) -> float:
    # Dot product
    dot_product = 0.0
    for x, y in zip(X, Y):
        dot_product += x * y

    # Magnitude of X
    magnitude_x = 0.0
    for x in X:
        magnitude_x += x * x
    magnitude_x = math.sqrt(magnitude_x)

    # Magnitude of Y
    magnitude_y = 0.0
    for y in Y:
        magnitude_y += y * y
    magnitude_y = math.sqrt(magnitude_y)

    # Avoid division by zero
    if magnitude_x == 0 or magnitude_y == 0:
        return 0.0

    return dot_product / (magnitude_x * magnitude_y)


# 2
def batch_embeddings(
    texts: list[str],
    embed_function: Callable[[list[str]], list[list[float]]],
    batch_size: int = 100,
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch = texts[start:end]

        embeddings = embed_function(batch)

        all_embeddings.extend(embeddings)

    return all_embeddings


# 3


def split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    tokens = sentence.split()

    chunks: list[str] = []

    for i in range(0, len(tokens), max_tokens):
        chunks.append(" ".join(tokens[i : i + max_tokens]))

    return chunks


def sentence_chunker(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 500,
) -> list[str]:
    sentences = nltk.tokenize(text)

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = sentence.split()
        sentence_length = len(sentence_tokens)

        # -----------------------------------------
        # Very long sentence → fixed-size fallback
        # -----------------------------------------
        if sentence_length > max_tokens:
            # First flush the current chunk
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            # Split the unusually long sentence
            long_chunks = split_long_sentence(sentence, max_tokens)

            chunks.extend(long_chunks)

            continue

        if current_tokens + sentence_length <= max_tokens:
            current_sentences.append(sentence)
            current_tokens += sentence_length

        else:
            chunks.append(" ".join(current_sentences))

            overlap_sentences: list[str] = []
            overlap_count = 0

            for previous_sentence in reversed(current_sentences):
                previous_length = len(previous_sentence.split())

                if overlap_count + previous_length > overlap_tokens:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_count += previous_length

            # Start new chunk with overlap
            current_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_count + sentence_length

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# 4
def deduplicate_chunks(chunks: list[str], threshold: float = 0.9) -> list[str]:
    unique: list[str] = []

    for chunk in chunks:
        words = set(chunk.lower().split())

        duplicate = False

        for existing in unique:
            existing_words = set(existing.lower().split())

            similarity = len(words & existing_words) / len(words | existing_words)

            if similarity >= threshold:
                duplicate = True
                break

        if not duplicate:
            unique.append(chunk)

    return unique


# 5
def check_dimensionality(X: list[float], Y: list[float]) -> bool:
    if len(X) != len(Y):
        raise ValueError(f"Embedding dimention noth match.{len(X)} != {len(Y)}")

    return True


# 6
class VectorIndex:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, text: str, embedding: list[float]) -> None:
        self.items.append({"text": text, "embedding": embedding})

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for item in self.items:
            embedding = item["embedding"]

            if len(query_embedding) != len(embedding):
                raise ValueError(
                    f"Embedding dimensions do not match: {len(query_embedding)} != {len(embedding)}"
                )

            score = cosine_similarity(query_embedding, item["embedding"])

            results.append({"text": item["text"], "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]


# 7


def embedding_cost(corpus_total_token: int, per_token_price: float) -> float:
    return corpus_total_token * per_token_price


# 8
def chunk_meta_data(text: str, source: str, chunk_size: int = 500) -> None:
    words = text.split()
    chunks: list[dict[str, Any]] = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]

        chunks.append(
            {
                "text": " ".join(chunk_words),
                "metadata": {
                    "source": source,
                    "chunk_index": len(chunks),
                    "start_index": i,
                    "end_index": i + len(chunk_words),
                },
            }
        )


# 9
def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = text.split()
    chunks: list[str] = []

    start = 0

    while start < len(tokens):
        end = start + chunk_size

        chunks.append(" ".join(tokens[start:end]))

        start += chunk_size - overlap

    return chunks


# 10
logger = logging.getLogger(__name__)


def reindex_corpus(
    corpus: list[str],
    index: VectorIndex,
    embed_function: Callable[[list[str], str], list[list[float]]],
    model_name: str,
) -> VectorIndex:
    """
    Re-embed and re-index the corpus using the new embedding model.

    corpus: list of text chunks
    index: vector index containing existing vectors
    embed_function: function(texts, model_name) -> embeddings
    model_name: currently configured embedding model
    """

    # Check for stale vectors
    for item in index.items:
        old_model = item.get("model")

        if old_model and old_model != model_name:
            logger.warning(
                "Stale embedding detected: stored model=%s, configured model=%s",
                old_model,
                model_name,
            )
            break

    # Re-embed entire corpus
    embeddings = embed_function(corpus, model_name)

    # Clear old vectors
    index.items.clear()

    # Add new vectors
    for text, embedding in zip(corpus, embeddings):
        index.items.append({"text": text, "embedding": embedding, "model": model_name})

    return index
