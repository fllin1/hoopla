import argparse

from hoopla.hybrid_search import (
    normalize_command,
    rrf_search_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser = subparsers.add_parser(
        "normalize", help="Compute min-max normalization"
    )
    norm_parser.add_argument(
        "scores", nargs="+", type=float, help="Add scores in the format '0.5 2.3 4 5.2'"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Compute weighted search"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="Query to search correspondance to"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Between 0.0 and 1.0 to weight exact matching and semantic search",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Limit of search results"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Uses the RRF formula to rank the results"
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="Query to search correspondance to"
    )
    rrf_search_parser.add_argument("--k", type=int, default=60, help="Weight constant")
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Limit of search results"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Apply reraking on top matches",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Use LLM to evaluate the relevancy of the matches",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.evaluate,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
