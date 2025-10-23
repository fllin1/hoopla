import argparse

from hoopla.hybrid_search import HybridSearch
from hoopla.utils import load_golden_dataset, load_movies


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden_dataset = load_golden_dataset()
    movie_dataset = load_movies()
    hybrid_search = HybridSearch(movie_dataset)

    print(f"k={limit}")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        results = hybrid_search.rrf_search(query, k=60, limit=limit)

        relevance_score = 0
        retrieved_titles = []
        for result in results:
            title = result["title"]
            retrieved_titles.append(title)
            if title in relevant_docs:
                relevance_score += 1
        precision = relevance_score / limit
        recall = relevance_score / len(relevant_docs)
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {query}")
        print(f"   - Precision@{limit}: {precision:.4f}")
        print(f"   - Recall@{limit}: {recall:.4f}")
        print(f"   - F1 Score: {f1_score:.4f}")
        print(f"   - Retrieved: {' ,'.join(retrieved_titles)}")
        print(f"   - Relevant: {' ,'.join(relevant_docs)}")


if __name__ == "__main__":
    main()
