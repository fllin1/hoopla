from hoopla.hybrid_search import HybridSearch
from hoopla.utils import call_gemini, load_movies


def augmented_generation_command(method: str, query: str, limit: int = 5):
    movie_dataset = load_movies()
    results = HybridSearch(movie_dataset).rrf_search(query, k=60, limit=limit)
    titles = [doc["title"] for doc in results]
    descriptions = [doc["document"] for doc in results]
    docs = [
        f"Title: {title}\nDescription: {description}"
        for title, description in zip(titles, descriptions)
    ]
    docs = "\n\n-".join(docs)
    rag_response = call_gemini(method=method, query=query, docs=docs)

    print("Search Results:")
    for doc in results:
        print(f"   - {doc['title']}")

    match method:
        case "rag":
            print("\nRAG Response:")
        case "summarize":
            print("\nLLM Summary:")
        case "citations":
            print("\nLLM Answer:")
        case "question":
            print("Answer")
    print(rag_response)
