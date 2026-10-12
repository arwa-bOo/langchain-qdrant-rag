# LangChain + Qdrant RAG on rag-mini-wikipedia

A small, measurable RAG pipeline: load → chunk → embed → Qdrant → LCEL chain → evaluate.
Dataset: [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) (CC-BY-3.0). Cost: $0 (free Colab + free Gemini key + in-memory Qdrant).

## Setup
1. Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com).
2. Open `langchain_qdrant_rag.ipynb` in Google Colab.
3. Run all cells and paste the key when prompted (or add it to Colab Secrets as `GOOGLE_API_KEY`). Qdrant runs in-memory (`:memory:`), nothing else to configure.
4. The run takes ~6-10 min because corpus upsert is batched with waits for the free-tier rate limit.

## Design
- 200 passages sampled with `seed=42`, split with `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` → 283 chunks.
- Embeddings: `gemini-embedding-001` (the specified `text-embedding-004` has been retired by Google). Collection `mini_wikipedia`, retriever `k=4`.
- LLM: `gemini-3.1-flash-lite`, auto-selected at runtime (the specified `gemini-2.0-flash` and `gemini-2.5-flash` are no longer available to new users). Prompt answers only from context, else "I don't know".
- 20 eval questions sampled with `seed=42`; `correct = 1` if lowercase ground truth is a substring of lowercase prediction. The metric was not changed after seeing results.

**Deviation:** the embedding dimension printed is **3072**, not 768. `gemini-embedding-001` returns 3072 dimensions by default and the `output_dimensionality=768` setting was not applied in the installed `langchain-google-genai` version. Qdrant sized the collection automatically, so retrieval still works.

## Results
**Accuracy: 3/20 = 0.15** (see `eval_results.csv`)

## Failure analysis
In 17 of 20 questions the model answered "I don't know". Most retrieved chunks were on the right general topic (Cleveland, Finland, Adams) but did not contain the answer, which is expected since only 200 of the 3,200 passages (~6%) were indexed, so many questions are effectively out-of-corpus; small chunks and k=4 also likely missed some relevant passages. Note the metric is also lenient in one case: two of the three "correct" rows (the "no" questions) match only because "no" is a substring of "I don't know", so the true accuracy is about 1/20.

## Files
- `langchain_qdrant_rag.ipynb` - the notebook (with outputs)
- `eval_results.csv` - 20 rows: question, retrieved_chunks, prediction, ground_truth, correct
