from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from hoopla.config import CACHE_DIR
from hoopla.utils import load_movies


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
        self.embeddings: Optional[NDArray] = None
        self.documents: Optional[List[dict]] = None
        self.document_map: Dict[int, dict] = {}

    def generate_embedding(self, text: str) -> NDArray:
        text = text.strip()
        if not text:
            raise ValueError("The input text is empty")
        embed = self.model.encode([text])
        return embed[0]

    def build_embeddings(self, documents: List[dict]) -> NDArray:
        self.documents = documents
        doc_str_repr: List[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_str_repr.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_str_repr, show_progress_bar=True)
        np.save(CACHE_DIR / "movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create(self, documents: List[dict]) -> NDArray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        cache_embed_path = CACHE_DIR / "movie_embeddings.npy"
        if cache_embed_path.exists():
            self.embeddings = np.load(cache_embed_path)
            if (self.embeddings is not None) and (
                self.embeddings.size == len(documents)
            ):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> List[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings or documents loaded. Call `load_or_create_embeddings` first."
            )
        embedding = self.generate_embedding(query)
        results = []
        for i in range(len(self.embeddings)):
            similarity_score = cosine_similarity(self.embeddings[i], embedding)
            results.append((similarity_score, self.documents[i]))
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return [
            {
                "score": result[0],
                "title": result[1]["title"],
                "description": result[1]["description"],
            }
            for result in results[:limit]
        ]


def verify_model_command() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text_command(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings_command() -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text_command(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity_command(vec1: NDArray, vec2: NDArray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def search_command(query: str, limit: int) -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create(documents)
    search_results = semantic_search.search(query, limit)
    for result in search_results:
        print(f"{result['title']}: (score: {result['score']})")
        description = result["description"]
        print(f"{description[: min(100, len(description))]} ...\n")
