import argparse
import mimetypes

from google.genai import types

from hoopla.utils import gemini_client


def main():
    parser = argparse.ArgumentParser(description="Describe an Image CLI")
    parser.add_argument("--query", type=str, help="Query")
    parser.add_argument("--image", type=str, help="Path to image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, mode="rb") as f:
        image = f.read()

    client = gemini_client()
    prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
  """
    parts = [
        prompt,
        types.Part.from_bytes(data=image, mime_type=mime),
        args.query.strip(),
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash-001", contents=parts
    )

    if response.text is None:
        raise ValueError("No output was given")
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
