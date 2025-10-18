import string
from typing import Dict, List

from nltk.stem import PorterStemmer

from hoopla.utils import load_movies, load_stopwords


def keyword_search(query: str) -> List[Dict[str, str]]:
    query_tokens = tokenize_text(query)

    movie_database = load_movies()
    movie_matches = []
    for movie in movie_database:
        movie_title = movie["title"]
        movie_title_tokens = tokenize_text(movie_title)

        for movie_title_token in movie_title_tokens:
            if any(query_token in movie_title_token for query_token in query_tokens):
                movie_matches.append(movie)
                break

    return sorted(movie_matches, key=lambda x: x["id"])


def tokenize_text(text: str) -> List[str]:
    punctuation_translator = str.maketrans({punc: "" for punc in string.punctuation})
    text = text.lower()
    text = text.translate(punctuation_translator)

    tokens = text.split()
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]
    return tokens
