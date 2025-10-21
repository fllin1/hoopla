#!/usr/bin/env python3

import argparse

from hoopla.inverted_index import InvertedIndex
from hoopla.keyword_search import tokenize_text
from hoopla.utils import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds cache data")

    tf_parser = subparsers.add_parser("tf", help="Look at term frequency for a doc id")
    tf_parser.add_argument("doc_id", type=int, help="Id of the document")
    tf_parser.add_argument("term", type=str, help="Should be single token to look for")

    idf_parser = subparsers.add_parser(
        "idf", help="Computes the inverse document frequency for a term"
    )
    idf_parser.add_argument("term", type=str, help="Should be a single token")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Computes the sum of TF and IDF"
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Look TF in the doc doc_id")
    tf_idf_parser.add_argument("term", type=str, help="Single token to compute TF-IDF")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Limit displayed results"
    )

    args = parser.parse_args()

    inv_idx = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            for query_token in tokenize_text(args.query):
                docs = inv_idx.get_documents(query_token)
                for i in range(min(len(docs), 5)):
                    movie_id = docs[i]
                    print(f"{i + 1}. {inv_idx.docmap[movie_id]['title']}")
        case "build":
            inv_idx.build()
            inv_idx.save()
        case "tf":
            inv_idx.load()
            tf = inv_idx.get_tf(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}': {tf}")
        case "idf":
            inv_idx.load()
            idf = inv_idx.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            inv_idx.load()
            tf_idf = inv_idx.get_tf(args.doc_id, args.term) * inv_idx.get_idf(args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            inv_idx.load()
            bm25_idf = inv_idx.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            inv_idx.load()
            bm25_tf = inv_idx.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}"
            )
        case "bm25search":
            inv_idx.load()
            bm25_search = inv_idx.bm25_search(args.query, args.limit)
            for idx, result in enumerate(bm25_search):
                print(
                    f"{idx + 1}. ({result['id']}) {result['title']} - Score: {result['score']:.2f}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
