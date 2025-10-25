from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

from hoopla.semantic_search import _cosine_similarity
from hoopla.utils import format_search_result, load_movies


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name, cache_folder="./models")
        self.documents = documents
        self.texts: list[str] = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        self.text_embeddings: NDArray = self.model.encode(self.texts)

    def embed_image(self, img_path: str) -> NDArray:
        image = Image.open(img_path)
        return self.model.encode(image, show_progress_bar=True)  # pyright: ignore

    def search_with_image(self, img_path: str) -> list[dict]:
        embedding = self.embed_image(img_path)
        results = []
        for i, text_embed in enumerate(self.text_embeddings):
            score = _cosine_similarity(embedding, text_embed)
            doc = self.documents[i]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                )
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]


def verify_image_embedding(img_path: str) -> None:
    multimodal_search = MultimodalSearch(documents=[])
    embedding = multimodal_search.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(img_path: str) -> None:
    movie_dataset = load_movies()
    multimodal_search = MultimodalSearch(movie_dataset)
    results = multimodal_search.search_with_image(img_path)

    for i, result in enumerate(results):
        print(f"\n{i + 1}. {result['title']} (similarity: {result['score']:.3f})")
        print(f"   {result['document'][:77]}...")
