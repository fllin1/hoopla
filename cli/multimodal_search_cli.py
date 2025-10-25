import argparse

from hoopla.multimodal_search import image_search_command, verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify that the image was correctly transformed to an embedding",
    )
    verify_image_parser.add_argument("img_path", type=str, help="Path to image")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search linked documents from image"
    )
    image_search_parser.add_argument("img_path", type=str, help="Path to image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.img_path)
        case "image_search":
            image_search_command(args.img_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
