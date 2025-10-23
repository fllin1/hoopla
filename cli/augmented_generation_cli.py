import argparse

from hoopla.augmented_generation import augmented_generation_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--limit", type=int, default=5, help="Limit of results")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform search + summary"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Limit of results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            augmented_generation_command("rag", args.query, args.limit)
        case "summarize":
            augmented_generation_command("summarize", args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
