import json
import string
from typing import Dict, List

from hoopla.config import DATA_DIR


def keyword_search(ref_title: str) -> List[Dict[str, str]]:
    punctuation_translator = str.maketrans({punc: "" for punc in string.punctuation})

    ref_title = ref_title.lower()
    ref_title = ref_title.translate(punctuation_translator)

    movie_matches = []
    with open(DATA_DIR / "movies.json", mode="r", encoding="utf-8") as f:
        movie_database: List[Dict[str, str]] = json.load(f)["movies"]

    for movie in movie_database:
        movie_title = movie["title"].lower()
        movie_title = movie_title.translate(punctuation_translator)

        if ref_title in movie_title:
            movie_matches.append(movie)

    return sorted(movie_matches, key=lambda x: x["id"])
