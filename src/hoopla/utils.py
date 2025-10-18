import json
from typing import Any, Dict, List, Set

from hoopla.config import DATA_DIR


def load_movies() -> List[Dict[str, Any]]:
    with open(DATA_DIR / "movies.json", mode="r", encoding="utf-8") as f:
        movie_database = json.load(f)["movies"]
    return movie_database


def load_stopwords() -> Set[str]:
    stopwords_path = DATA_DIR / "stopwords.txt"
    stopwords = stopwords_path.read_text(encoding="utf-8").splitlines()
    return set(stopwords)
