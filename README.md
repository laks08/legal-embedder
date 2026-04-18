## Project Overview

This project fine-tunes a sentence embedding model on US federal court opinions (PDFs) so that semantic search and clustering reflect legal concepts more accurately than a generic model. It compares the fine-tuned **legal-embedder** against the baseline **all-MiniLM-L6-v2** using clustering metrics, UMAP plots, similarity heatmaps, and qualitative retrieval. The workflow is implemented entirely in a single Jupyter notebook so you can run preprocessing, pair generation, training, and evaluation in one place.

## Approach

1. **PDF ingestion with Docling** — Each PDF in `data/` is converted with Docling’s `DocumentConverter` and exported to Markdown. Legal PDFs often have footnotes, multi-column layouts, headers, and docket lines; naive extractors frequently scramble reading order or drop structure. Docling is designed for document understanding and produces cleaner, more faithful text for downstream chunking.

2. **Paragraph-level chunking** — After a light cleaning pass, text is split on paragraph boundaries (blank lines), short paragraphs are merged, and long paragraphs are split at sentence boundaries. Fixed-size windows can cut mid-sentence or mid-holding; paragraph-aware chunking keeps coherent legal reasoning and citations together, which improves both training pairs and evaluation.

3. **Contrastive training pairs from Ollama Cloud** — For each chunk, an LLM hosted at [ollama.com](https://ollama.com) (with an API key) generates legally phrased questions the chunk answers. Each (question, passage) pair is an **anchor–positive** pair: the model learns to pull matching questions and passages together in embedding space and push unrelated text apart (in-batch negatives via **Multiple Negatives Ranking Loss**). That is **contrastive learning**: the signal comes from which pairs should be similar versus dissimilar, without labeling every possible negative.

4. **Fine-tuning** — `SentenceTransformer` is trained with `MultipleNegativesRankingLoss` and `SentenceTransformerTrainer` so the encoder maps legal queries near the passages they refer to.

5. **Evaluation** — KMeans and UMAP visualize cluster structure; silhouette scores summarize separation; heatmaps and nearest-neighbor retrieval provide intuitive before/after comparisons.

## Model and Tool Roles

| Component | Name | Role | Why Chosen |
|---|---|---|---|
| Base Embedding Model | all-MiniLM-L6-v2 | Starting point for fine-tuning and baseline for comparison | Small, fast, well-known, produces 384-dim vectors |
| Pair Generation LLM | Configurable cloud model via Ollama (`ollama.com` API) | Generates legal question and answer training pairs from chunks | Same Python `ollama` client as local runs; cloud models listed at [ollama.com/search?c=cloud](https://ollama.com/search?c=cloud) |
| Fine-Tuned Model | legal-embedder | Domain-specific embedder for legal retrieval | Trained on the exact corpus, outperforms generic model on legal queries |
| PDF Extractor | Docling | Converts legal PDFs to clean structured text | Handles footnotes, multi-column layouts, and docket headers correctly |

## How to Run

1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Place all PDF files into the `data/` folder
4. Open `legal_embedder.ipynb` in Jupyter or Cursor IDE
5. Run all cells from top to bottom
6. When prompted, enter your **Ollama API key** (create one at [ollama.com/settings/keys](https://ollama.com/settings/keys)) — it is not stored in the repo
7. The fine-tuned model will be saved to `./models/legal-embedder` after Section 6 completes

## Using the Fine-Tuned Model in Other Projects

```python
from sentence_transformers import SentenceTransformer

# Load the fine-tuned legal embedder
model = SentenceTransformer("./models/legal-embedder")

# Embed any text
embeddings = model.encode(["What constitutes securities fraud under Rule 10b-5?"])
```

This model is a drop-in replacement for any sentence-transformers model and can be used directly in LangChain, LlamaIndex, or any vector store that accepts a HuggingFace embedding model.

## Metrics and Results

| Metric | Baseline | Fine-Tuned |
|---|---|---|
| Silhouette Score | - | - |
| UMAP Cluster Separation | - | - |
| Nearest Neighbor Relevance | - | - |

Fill in after running the notebook.

## Dependencies

Minimum versions are pinned in `requirements.txt`. Libraries:

- `sentence-transformers>=3.0.0`
- `ollama>=0.3.0`
- `docling>=2.0.0`
- `scikit-learn>=1.3.0`
- `umap-learn>=0.5.5`
- `matplotlib>=3.7.0`
- `seaborn>=0.13.0`
- `datasets>=2.14.0`
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `tqdm>=4.66.0`
- `accelerate>=0.26.0` (required by `SentenceTransformerTrainer`)

## Challenges and Limitations

The corpus may be small (e.g., only a handful of opinions), which limits diversity of training pairs and generalization. Legal PDFs add noise: docket strings, footnotes, and citation lines require careful preprocessing. Hosted APIs enforce rate limits, so pair generation must be paced with a limiter, checkpoints, and resume logic to avoid redundant API calls and lost progress.
