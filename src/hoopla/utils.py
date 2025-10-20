import json
from typing import Any, Dict, List, Set

from hoopla.config import DATA_DIR

BM25_K1 = 1.5
BM25_B = 0.75

SCORE_PRECISION = 3


def load_movies() -> List[Dict[str, Any]]:
    with open(DATA_DIR / "movies.json", mode="r", encoding="utf-8") as f:
        movie_database = json.load(f)["movies"]
    return movie_database


def load_stopwords() -> Set[str]:
    stopwords_path = DATA_DIR / "stopwords.txt"
    stopwords = stopwords_path.read_text(encoding="utf-8").splitlines()
    return set(stopwords)


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
