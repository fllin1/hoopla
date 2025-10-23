import math
import pickle
from collections import Counter
from typing import Any, Callable

from hoopla.config import CACHE_DIR
from hoopla.keyword_search import tokenize_text
from hoopla.utils import BM25_B, BM25_K1, format_search_result, load_movies


def text_single_token(text: str) -> str:
    tokens = tokenize_text(text)
    if len(tokens) != 1:
        raise Exception("there should be a single token in term")
    return tokens[0]


class InvertedIndex:
    """
    An inverted index is what makes search fast – it's like a SQL database index, but for text search.
    Instead of scanning every document each time a user searches, we build an index for fast lookups.
    """

    def __init__(self):
        """
        A "forward index" maps location -> value. (docmap)
        An "inverted index" maps value -> location. (index)
        """
        self.docmap: dict[int, dict[str, Any]] = {}
        self.index: dict[str, set[int]] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def __compute_idf_function(
        self, token: str, func: Callable[[int, int], float]
    ) -> float:
        if token not in self.index:
            return 0
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return func(doc_count, term_doc_count)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        total_doc_length = 0
        for length in self.doc_lengths.values():
            total_doc_length += length
        return total_doc_length / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        if term not in self.index:
            return []
        doc_ids: set[int] = self.index[term.strip().lower()]
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = text_single_token(term)
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        token = text_single_token(term)
        return self.__compute_idf_function(
            token, lambda x, y: math.log((x + 1) / (y + 1))
        )

    def get_bm25_idf(self, term: str) -> float:
        token = text_single_token(term)
        return self.__compute_idf_function(
            token, lambda x, y: math.log((x - y + 0.5) / (y + 0.5) + 1)
        )

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)

    def bm25_search(self, query: str, limit: int) -> list[dict]:
        tokens = tokenize_text(query)
        scores = {}
        for token in tokens:
            if token not in self.index:
                continue
            doc_ids = self.index[token]
            for doc_id in doc_ids:
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += self.bm25(doc_id, token)
        results = []
        for id, score in scores.items():
            document = self.docmap[id]
            results.append(
                format_search_result(
                    doc_id=id,
                    title=document["title"],
                    document=document["description"],
                    score=score,
                )
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

    def build(self) -> None:
        movies_database = load_movies()
        for movie in movies_database:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        with open(CACHE_DIR / "index.pkl", mode="wb") as f:
            pickle.dump(self.index, f)
        with open(CACHE_DIR / "docmap.pkl", mode="wb") as f:
            pickle.dump(self.docmap, f)
        with open(CACHE_DIR / "term_frequencies.pkl", mode="wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(CACHE_DIR / "doc_lengths.pkl", mode="wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(CACHE_DIR / "index.pkl", mode="rb") as f:
            self.index = pickle.load(f)
        with open(CACHE_DIR / "docmap.pkl", mode="rb") as f:
            self.docmap = pickle.load(f)
        with open(CACHE_DIR / "term_frequencies.pkl", mode="rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(CACHE_DIR / "doc_lengths.pkl", mode="rb") as f:
            self.doc_lengths = pickle.load(f)
