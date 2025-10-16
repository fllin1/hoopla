#!/usr/bin/env python3

import argparse

from hoopla.keyword_search import keyword_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movie_matches = keyword_search(args.query)
            for i in range(min(len(movie_matches), 5)):
                print(f"{i + 1}. {movie_matches[i]['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
