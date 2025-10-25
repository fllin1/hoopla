# Hoopla

Modern search engine techniques implemented end-to-end: keyword (BM25), semantic (embeddings), hybrid fusion (weighted, RRF), multimodal (image→text), and LLM-enhanced retrieval (RAG with Gemini). Includes CLIs, caching, and evaluation utilities.

## Features
- **Keyword search (BM25)**: Tokenization, stemming, stopwording, inverted index with TF/IDF, BM25, caching.
- **Semantic search**: SentenceTransformers embeddings, cosine similarity, optional description chunking.
- **Hybrid search**: Fuse BM25 and semantic via weighted sum or RRF; optional reranking (LLM or cross-encoder).
- **RAG**: Retrieve with hybrid search and generate answers via Gemini using prompt templates.
- **Multimodal**: CLIP-based image embedding to retrieve related movies.
- **Evaluation**: Precision/Recall/F1 against a golden dataset.

## Quickstart
```bash
# Clone and enter
git clone https://github.com/fllin1/hoopla
cd hoopla/

# Setup using uv (creates .venv)
uv sync
uv pip install -e .

# Activate (Linux)
source .venv/bin/activate
```

### Environment
- Create a `.env` in the project root with:
```
GEMINI_API_KEY=your_api_key_here
```

### Data
Place the following files under `data/`:
- `movies.json` – movie corpus
- `stopwords.txt` – stopword list
- `golden_dataset.json` – evaluation queries and relevant docs

You will be able to find all the data files on this [Mega Folder](https://mega.nz/folder/qY8UnZpL#nNuUQnyFtIl4bfKMWdCAnA). 

Directory convenience paths are created automatically for `cache/` generated indices/embeddings

## Command Line Interfaces
All CLIs live in `cli/`. Run with `python cli/<name>.py <command> ...`.

### Keyword & BM25 – `keyword_search_cli.py`
- Build and cache inverted index:
```bash
python cli/keyword_search_cli.py build
```
- Inspect/search:
```bash
python cli/keyword_search_cli.py search "hobbit"
python cli/keyword_search_cli.py tf 42 "ring"
python cli/keyword_search_cli.py idf "ring"
python cli/keyword_search_cli.py tfidf 42 "ring"
python cli/keyword_search_cli.py bm25idf "ring"
python cli/keyword_search_cli.py bm25tf 42 "ring" 1.5 0.75
python cli/keyword_search_cli.py bm25search "journey to mount doom" 5
```

### Semantic – `semantic_search_cli.py`
```bash
python cli/semantic_search_cli.py verify
python cli/semantic_search_cli.py embed_text "a sample sentence"
python cli/semantic_search_cli.py verify_embeddings
python cli/semantic_search_cli.py embedquery "elves and dwarves alliance"
python cli/semantic_search_cli.py search "one ring" --limit 5
python cli/semantic_search_cli.py chunk "Long text here" --chunk-size 200 --overlap 0
python cli/semantic_search_cli.py semantic_chunk "Sentences here." --max-chunk-size 4 --overlap 1
python cli/semantic_search_cli.py embed_chunks
python cli/semantic_search_cli.py search_chunked "dark wizard" --limit 5
```

### Hybrid – `hybrid_search_cli.py`
```bash
python cli/hybrid_search_cli.py normalize 0.1 0.5 0.9
python cli/hybrid_search_cli.py weighted-search "dragon quest" --alpha 0.5 --limit 5
python cli/hybrid_search_cli.py rrf-search "ring fellowship" --k 60 --limit 5 \
  --enhance spell --rerank-method cross_encoder --evaluate
```
- Rerank methods:
  - `individual` and `batch` use Gemini via prompt templates
  - `cross_encoder` uses `cross-encoder/ms-marco-TinyBERT-L2-v2`

### RAG – `augmented_generation_cli.py`
```bash
python cli/augmented_generation_cli.py rag "Who is the ring-bearer?" --limit 5
python cli/augmented_generation_cli.py summarize "Plot of the first journey" --limit 5
python cli/augmented_generation_cli.py citations "Who forged the ring?" --limit 5
python cli/augmented_generation_cli.py question "What is Mordor?" --limit 5
```

### Multimodal – `multimodal_search_cli.py`
```bash
python cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
python cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

### Evaluation – `evaluation_cli.py`
```bash
python cli/evaluation_cli.py --limit 5
```

## Modules Overview
- `hoopla.keyword_search`: tokenization with stemming/stopwords; basic title matching.
- `hoopla.inverted_index`: index build/load/save, TF/IDF/BM25, BM25 search.
- `hoopla.semantic_search`: embedding model, doc/chunk embeddings, cosine search.
- `hoopla.hybrid_search`: score normalization, weighted or RRF fusion, optional reranking and LLM-based evaluation.
- `hoopla.augmented_generation`: RAG orchestration over hybrid results + Gemini.
- `hoopla.multimodal_search`: CLIP text/image embeddings for image→text retrieval.
- `hoopla.utils`: IO helpers, Gemini client, prompt templating, formatting.
- `hoopla.config`: paths for `data/`, `cache/`, `cli/`.

## Caching & Models
- Caches under `cache/`: indices, embeddings, metadata.
- HuggingFace models are cached under `./models` (checked into repo directories).
  - Sentence embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - Cross-encoder: `cross-encoder/ms-marco-TinyBERT-L2-v2`
  - Multimodal: `sentence-transformers/clip-ViT-B-32`

## Notes & Tips
- Ensure `GEMINI_API_KEY` is set for any command that contacts Gemini.
- First runs may download models; subsequent runs use local cache.
- If you change `movies.json`, clear `cache/` to regenerate indices/embeddings.

## License
MIT
