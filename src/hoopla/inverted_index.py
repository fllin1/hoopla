import pickle
from typing import Any, Dict, List, Set

from hoopla.config import PROJECT_ROOT
from hoopla.keyword_search import tokenize_text
from hoopla.utils import load_movies


class InvertedIndex:
    """
    An inverted index is what makes search fast – it's like a SQL database index, but for text search.
    Instead of scanning every document each time a user searches, we build an index for fast lookups.
    """

    def __init__(self):
        """
        A "forward index" maps location -> value.
        An "inverted index" maps value -> location.
        """
        self.docmap: Dict[int, Dict[str, Any]] = {}
        self.index: Dict[str, Set[int]] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

    def get_documents(self, term: str) -> List[int]:
        if term not in self.index:
            return []
        doc_ids: Set[int] = self.index[term.strip().lower()]
        return sorted(list(doc_ids))

    def build(self) -> None:
        """
        Using an inverted index is super fast – we get O(1) lookups on each token.
        However, building the index is slow, because we have to read every document and tokenize it.
        """
        movies_database = load_movies()
        for movie in movies_database:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        cache_dir = (PROJECT_ROOT / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "index.pkl", mode="wb") as f:
            pickle.dump(self.index, f)
        with open(cache_dir / "docmap.pkl", mode="wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        cache_dir = (PROJECT_ROOT / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "index.pkl", mode="rb") as f:
            self.index = pickle.load(f)
        with open(cache_dir / "docmap.pkl", mode="rb") as f:
            self.docmap = pickle.load(f)
