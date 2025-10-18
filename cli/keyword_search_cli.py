#!/usr/bin/env python3

import argparse
import sys

from hoopla.inverted_index import InvertedIndex
from hoopla.keyword_search import tokenize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds cache data")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            inv_idx = InvertedIndex()
            try:
                inv_idx.load()
            except FileNotFoundError:
                print("inverted index and docmap not cached")
                sys.exit(1)
            for query_token in tokenize_text(args.query):
                docs = inv_idx.get_documents(query_token)
                for i in range(min(len(docs), 5)):
                    movie_id = docs[i]
                    print(f"{i + 1}. {inv_idx.docmap[movie_id]['title']}")
        case "build":
            inv_idx = InvertedIndex()
            inv_idx.build()
            inv_idx.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
