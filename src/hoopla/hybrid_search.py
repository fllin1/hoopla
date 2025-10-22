from typing import Optional

from hoopla.config import CACHE_DIR
from hoopla.inverted_index import InvertedIndex
from hoopla.semantic_search import ChunkedSemanticSearch
from hoopla.utils import call_gemini, format_search_result, load_movies


def normalize(scores: list[float]) -> list[float]:
    max_score, min_score = max(scores), min(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    spread = max_score - min_score
    return [(score - min_score) / spread for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not (CACHE_DIR / "index.pkl").exists():
            self.idx.build()
            self.idx.save()

    def __get_keyword_and_semantic_scores(
        self,
        keyword_results: list[dict],
        semantic_results: list[dict],
    ) -> dict[int, dict]:
        keyword_scores = [x["score"] for x in keyword_results]
        keyword_idx = [x["id"] for x in keyword_results]
        semantic_scores = [x["score"] for x in semantic_results]
        semantic_idx = [x["id"] for x in semantic_results]

        norm_keyword_scores = normalize(keyword_scores)
        norm_semantic_scores = normalize(semantic_scores)
        norm_keyword_results = {
            idx: score for idx, score in zip(keyword_idx, norm_keyword_scores)
        }
        norm_semantic_results = {
            idx: score for idx, score in zip(semantic_idx, norm_semantic_scores)
        }
        scores_map: dict[int, dict[str, float]] = {}

        rank = 0
        for idx, score in norm_keyword_results.items():
            rank += 1
            if idx not in semantic_idx:
                continue
            scores_map[idx] = {"keyword_score": score, "keyword_rank": rank}

        rank = 0
        for idx, score in norm_semantic_results.items():
            rank += 1
            if idx not in scores_map:
                continue
            scores_map[idx]["semantic_score"] = score
            scores_map[idx]["semantic_rank"] = rank
        return scores_map

    def __format_scores_map(
        self, scores_map: dict[int, dict[str, float]]
    ) -> list[dict]:
        hybrid_results = []
        for id, score_map in scores_map.items():
            document = self.idx.docmap[id]
            hybrid_results.append(
                format_search_result(
                    doc_id=str(id),
                    title=document["title"],
                    document=document["description"],
                    score=score_map["score"],
                    keyword_score=score_map["keyword_score"],
                    semantic_score=score_map["semantic_score"],
                )
            )
        return hybrid_results

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        scores_map = self.__get_keyword_and_semantic_scores(
            bm25_results, semantic_results
        )
        for _, scores in scores_map.items():
            scores["score"] = hybrid_score(
                scores["keyword_score"], scores["semantic_score"], alpha
            )
        hybrid_results = self.__format_scores_map(scores_map)
        return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        scores_map = self.__get_keyword_and_semantic_scores(
            bm25_results, semantic_results
        )
        for _, scores in scores_map.items():
            scores["score"] = rrf_score(scores["keyword_rank"], k) + rrf_score(
                scores["semantic_rank"], k
            )
        hybrid_results = self.__format_scores_map(scores_map)
        return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)[:limit]


def normalize_command(scores: list[float]) -> None:
    scores = normalize(scores)
    for score in scores:
        print(f"* {score:.4f}")


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    scores = hybrid_search.weighted_search(query, alpha, limit)
    for i, result in enumerate(scores):
        metadata = result["metadata"]
        print(f"{i + 1}. {result['title']}")
        print(f"   Hybrid Score: {result['score']:.3f}")
        print(
            f"   BM25: {metadata['keyword_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
        )
        print(f"   {result['document'][:77]}...")


def rrf_search_command(
    query: str, k: int, limit: int, enhance: Optional[str] = None
) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    if enhance is not None:
        old_query = query
        query = call_gemini(old_query, enhance)
        print(f"Enhanced query ({enhance}): '{old_query}' -> '{query}'")

    scores = hybrid_search.rrf_search(query, k, limit)
    for i, result in enumerate(scores):
        metadata = result["metadata"]
        print(f"{i + 1}. {result['title']}")
        print(f"   RRF score: {result['score']:.3f}")
        print(
            f"   BM25: {metadata['keyword_score']:.3f}, semantic: {metadata['semantic_score']:.3f}"
        )
        print(f"   {result['document'][:77]}...")
