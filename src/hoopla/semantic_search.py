import json
import re
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from hoopla.config import CACHE_DIR
from hoopla.utils import format_search_result, load_movies


def _cosine_similarity(vec1: NDArray, vec2: NDArray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def _chunking(
    text: str, chunk_size: int, overlap: int, split_func: Callable[[str], List[str]]
) -> List[str]:
    assert overlap >= 0, "Overlap takes positive values"
    assert overlap < chunk_size, "Overlap should be bigger than chunk_size"
    raw_chunks = split_func(text)
    output_chunks: List[str] = []
    i = 0
    while i < len(raw_chunks) - overlap:
        formatted_chunk = " ".join(raw_chunks[i : i + chunk_size]).strip()
        if len(formatted_chunk) > 0:
            output_chunks.append(formatted_chunk)
        i += chunk_size - overlap
    return output_chunks


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name, cache_folder="./models")
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
            similarity_score = _cosine_similarity(self.embeddings[i], embedding)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: Optional[NDArray] = None
        self.chunk_metadata: Optional[List[dict]] = None

    def build_chunk_embeddings(self, documents: List[dict]) -> NDArray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        all_chunks: List[str] = []
        chunk_metadata = []
        for doc in documents:
            description = doc["description"].strip()
            if len(description) == 0:
                continue
            desc_chunks = _chunking(
                description, 4, 1, lambda x: re.split(r"(?<=[.!?])\s+", x)
            )
            all_chunks.extend(desc_chunks)
            chunk_metadata.extend(
                {
                    "movie_idx": doc["id"],
                    "chunk_idx": i,
                    "total_chunks": len(desc_chunks),
                }
                for i in range(len(desc_chunks))
            )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = {
            "chunks": chunk_metadata,
            "total_chunks": len(all_chunks),
        }
        np.save(CACHE_DIR / "chunk_embeddings.npy", self.chunk_embeddings)
        with open(CACHE_DIR / "chunk_metadata.json", mode="w", encoding="utf-8") as f:
            json.dump(self.chunk_metadata, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: List[dict]) -> NDArray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        path_chunk_embed = CACHE_DIR / "chunk_embeddings.npy"
        path_chunk_meta = CACHE_DIR / "chunk_metadata.json"
        if path_chunk_embed.exists() and path_chunk_meta.exists():
            self.chunk_embeddings: NDArray = np.load(path_chunk_embed)
            with open(path_chunk_meta, mode="r", encoding="utf-8") as f:
                self.chunk_metadata: dict = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> List[dict]:
        query_embed = self.generate_embedding(query)
        chunk_scores = []
        chunk_metadata = self.chunk_metadata["chunks"]
        for chunk_idx, chunk_embed in enumerate(self.chunk_embeddings):
            score = _cosine_similarity(query_embed, chunk_embed)
            chunk_scores.append(
                {
                    "chunk_idx": chunk_idx,
                    "movie_idx": chunk_metadata[chunk_idx]["movie_idx"],
                    "score": score,
                }
            )
        best_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            current_score = chunk_score["score"]
            if (movie_idx not in best_scores) or (
                best_scores[movie_idx] < current_score
            ):
                best_scores[movie_idx] = current_score
        best_scores = dict(
            sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        )
        results = []
        for movie_idx, score in best_scores.items():
            document = self.document_map[movie_idx]
            results.append(
                format_search_result(
                    movie_idx, document["title"], document["description"][:100], score
                )
            )
        return results


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


def search_command(query: str, limit: int) -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create(documents)
    search_results = semantic_search.search(query, limit)
    for i, result in enumerate(search_results):
        print(f"{i + 1}. {result['title']}: (score: {result['score']})")
        description = result["description"]
        print(f"   {description[: len(description)]} ...\n")


def chunk_command(text: str, chunk_size: int, overlap: int) -> None:
    output_chunks = _chunking(text, chunk_size, overlap, lambda x: x.split())
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(output_chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunk_command(text: str, chunk_size: int, overlap: int) -> None:
    output_chunks = _chunking(
        text, chunk_size, overlap, lambda x: re.split(r"(?<=[.!?])\s+", x)
    )
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(output_chunks):
        print(f"{i + 1}. {chunk}")


def embed_chunks_command() -> None:
    documents = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int) -> None:
    documents = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    results = chunked_semantic_search.search_chunks(query, limit=limit)
    for i, document in enumerate(results):
        print(f"\n{i + 1}. {document['title']} (score: {document['score']:.4f})")
        print(f"   {document['document']}...")
