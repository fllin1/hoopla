#!/usr/bin/env python3

import argparse

from hoopla.semantic_search import (
    embed_query_text_command,
    embed_text_command,
    search_command,
    verify_embeddings_command,
    verify_model_command,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify loaded model and its max seq length")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Tests embedding generation"
    )
    embed_text_parser.add_argument("text", help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify load and save embeddings")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Tests embedding of queries"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search top cosine similarity for a query"
    )
    search_parser.add_argument(
        "query", type=str, help="Query to compare similarity with docs"
    )
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Limit of results displayed"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model_command()
        case "embed_text":
            embed_text_command(args.text)
        case "verify_embeddings":
            verify_embeddings_command()
        case "embedquery":
            embed_query_text_command(args.query)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
