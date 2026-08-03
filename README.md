# RAG Hallucination Reduction — Research Project

MSc Advanced Software Engineering Dissertation
K.G. Sachintha Udara | University of Westminster via IIT

Compares three RAG configurations on the same Azure OpenAI model to measure how
retrieval and a post-generation **Grounded Response Validator (GRV)** affect
hallucination rate:

| Config | Description |
|--------|-------------|
| **A** | Baseline LLM, no retrieval |
| **B** | RAG — hybrid retrieval (Azure AI Search) + generation |
| **C** | RAG + GRV — Config B's answer is scored for groundedness by a 3-layer validator |

---

## 1. Prerequisites

Azure resources (all provisioned under one resource group):

- **Azure OpenAI / AI Foundry** — a chat deployment (e.g. `gpt-4.1-mini`) and an
  embedding deployment (e.g. `text-embedding-3-large`)
- **Azure AI Search** — for the vector/hybrid index, with semantic ranking enabled
  (a semantic configuration named `default`)
- **Azure Blob Storage** — hosts the raw corpus `.md` files; indexed chunks link
  back to their source blob via a SAS URL
- **Azure Table Storage** — experiment logging (every pipeline call writes a row)

Local: Python 3.11, `venv`.

---

## 2. Setup from scratch

```bash
git clone <repo-url>
cd rag-project

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.template .env
# then edit .env with your Azure endpoints/keys (see below)
```

`.env` fields (see `.env.template`):

```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_GPT4O_DEPLOYMENT=gpt-4.1-mini-deployment
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large-deployment
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=rag-documents-index
AZURE_STORAGE_CONNECTION_STRING=
AZURE_BLOB_CONTAINER=documents
AZURE_TABLE_NAME=experimentlogs

GRV_THRESHOLD=0.6
TOP_K_CHUNKS=5
```

The corpus itself is already checked into `data/corpus/` (160 Markdown docs
across two topics: `azure-functions/`, `container-apps/`) — no download step
needed unless you're replacing it with a different topic set.

---

## 3. Build the index and load data

### 3.1 Upload the corpus to Blob Storage

```bash
python scripts/upload_to_blob.py
```

Creates the `documents` container (if missing) and uploads every `.md` file
under `data/corpus/<topic>/` as `<topic>/<filename>.md`. This is the
canonical copy the index's `blob_url` field points back to.

### 3.2 Chunk and index the corpus into Azure AI Search

```bash
python -m src.ingestion.run_ingestion
```

This:
1. Loads all `.md` files from `CORPUS_FOLDERS` in `src/ingestion/run_ingestion.py`
2. Splits them with `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=1000`,
   `CHUNK_OVERLAP=200`, `src/ingestion/chunker.py`)
3. Embeds each chunk (`text-embedding-3-large`) and pushes it into
   `AZURE_SEARCH_INDEX_NAME` in batches of 100 (`src/ingestion/indexer.py`)

Batch progress is checkpointed to `logs/ingestion_progress.json` — if a batch
fails (e.g. rate limit) just re-run the same command and it resumes from the
last completed batch. On a clean finish this produced 2,955 chunks from 160
documents.

### 3.3 Tag indexed chunks with their blob URL

The search index doesn't know about Blob Storage until you patch it. Run
these once, in order, after the first successful ingestion:

```bash
# add a blob_url field to the index schema (no-op if it already exists)
python scripts/add_blob_url_field.py

# backfill blob_url on every existing document, derived from its metadata.source path
python scripts/patch_blob_urls_full.py

# spot-check a few documents to confirm the field was patched correctly
python scripts/verify_blob_urls.py
```

`retrieve_chunks()`/`get_chunk_sources()` in `src/pipeline/retriever.py` use
this field (plus a short-lived SAS token) so answers can cite a viewable
source link.

---

## 4. Calibrate the abstention threshold (ground-truth value from the current corpus)

Config C's GRV has two scoring paths: the standard 3-layer score, and an
**abstention-verification** path used when the model refuses to answer
(`src/validator/abstention.py` pattern-matches refusal phrasing). The
abstention path needs a calibrated relevance threshold `tau` — the "true"
cut-off computed empirically against *this* corpus/index, not a hardcoded
guess.

```bash
python scripts/calibrate_abstention_threshold.py
```

This queries the **live index** (via semantic search,
`src/pipeline/retriever_expanded.py`) for every question in
`data/questions/qa_dataset.json` marked `in_corpus: true`, takes each
question's max reranker relevance score, and sets `tau` to the 10th
percentile of that distribution. Result is written to
`data/abstention_threshold.json`:

```json
{
  "tau": 2.8514,
  "percentile_used": 10,
  "n_calibration_questions": 140,
  "score_distribution": { "min": 2.51, "p10": 2.85, "median": 3.19, "max": 3.67 }
}
```

`src/validator/abstention_verify.py` reads this file at runtime. **Re-run
this step any time the corpus or index changes** — `tau` is only valid for
the corpus it was calibrated against.

### The QA ground-truth dataset

`data/questions/qa_dataset.json` (200 questions) is the labelled ground truth
the pipelines are evaluated against — 150 in-corpus questions (each with an
`expected_answer`, `source_doc`, and `source_passage` pulled from the actual
corpus text) and 50 out-of-corpus questions (`in_corpus: false`, no expected
answer) used to test correct abstention. `data/questions/dev_test.json` is a
10-question smoke-test subset for quick manual checks. Both already exist in
the repo; if you swap in a different corpus you'll need to re-derive this
file's `source_doc`/`source_passage`/`expected_answer` fields from the new
documents (there is currently no automated generator script for this step —
it was done as a one-off curation pass and the batch-generation scripts were
removed after use, see `git log` on `data/questions/`).

---

## 5. Try it — compare all three configs on demo questions

```bash
python scripts/demo_compare_configs.py
```

Runs three fixed questions (two in-corpus, one out-of-corpus/off-topic)
through Config A, B, and C side by side in the terminal, with colour-coded
output showing each config's answer, retrieved source links, GRV score,
label (`grounded` / `partially_grounded` / `ungrounded`), scoring path, and
latency breakdown. This is the fastest way to confirm the whole stack
(index, blob tagging, GRV layers, Table Storage logging) is wired up
correctly after setup.

Edit `DEMO_QUESTIONS` at the top of the script to try your own questions.

---

## 6. Run the API

```bash
uvicorn main:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create an Azure Function trigger?", "config": "C"}'
```

`config` is `"A"`, `"B"`, or `"C"`. Every call is logged to
`AZURE_TABLE_NAME` via `src/utils/logger.py`.

---

## 7. Tests

```bash
python -m pytest tests/ -v
```

`tests/test_pipeline.py` exercises all three configs end-to-end (live Azure
calls) plus two unit checks on `src/validator/grv.py` using hand-written
grounded/hallucinated chunk pairs — useful as a fast sanity check that
credentials and deployments are all correct before a full evaluation run.

---

## 8. Full evaluation and metrics

`results/results_config_{A,B,C}.csv` hold the dissertation's evaluation
results — each config run over the full `qa_dataset.json`. There's no bundled
batch-runner script (the original one was a one-off and was removed after
producing these results); to regenerate, loop over
`data/questions/qa_dataset.json` calling `config_a.query()` /
`config_b.query()` / `config_c.query()` per question and collect the
returned dicts into a CSV — that's exactly what each config's `query()`
function returns already.

Once results exist:

```bash
# human annotation: data/annotation_template.csv (question_id, human_label, notes)
# fill in human_label per question_id for a sample of Config C responses first

cd scripts
python calculate_metrics.py       # hallucination rate per config, Cohen's Kappa vs human labels,
                                   # per-category breakdown, RAGAS summary -> ../logs/metrics_summary.json
python compute_final_stats.py     # latency aggregation + per-layer GRV score stats
python show_kappa_calculation.py  # step-by-step worked Cohen's Kappa calculation (dissertation appendix)
```

`calculate_metrics.py` and `compute_final_stats.py` use relative paths and
must be run **from inside `scripts/`**.

---

## Project structure

```
src/
  config/       Settings loaded from .env
  ingestion/    Chunking + Azure AI Search indexing
  pipeline/     config_a.py / config_b.py / config_c.py + retriever.py, retriever_expanded.py
  validator/    GRV: layer_crossencoder.py, layer_ragas.py, layer_reranker.py, grv.py,
                abstention.py, abstention_verify.py
  utils/        Table Storage logger, timer

scripts/
  upload_to_blob.py               Upload corpus to Blob Storage
  add_blob_url_field.py           Add blob_url field to the search index schema
  patch_blob_urls_full.py         Backfill blob_url on indexed documents
  verify_blob_urls.py             Spot-check patched blob_url values
  calibrate_abstention_threshold.py  Compute tau against the current corpus/index
  demo_compare_configs.py         Side-by-side demo of Config A/B/C
  calculate_metrics.py            Hallucination rate, Cohen's Kappa, category/RAGAS summaries
  compute_final_stats.py          Latency + per-layer GRV score stats
  show_kappa_calculation.py       Worked Cohen's Kappa calculation

data/
  corpus/azure-functions/, corpus/container-apps/   160 source .md docs
  questions/qa_dataset.json       200 ground-truth QA pairs (150 in-corpus, 50 out-of-corpus)
  questions/dev_test.json         10-question smoke-test subset
  annotation_template.csv         Human annotation sheet (question_id, human_label, notes)
  abstention_threshold.json       Calibrated tau (generated by step 4 above)

results/        Evaluation output CSVs per config
logs/           metrics_summary.json + ingestion progress checkpoint
```

## GRV scoring (Config C)

Three layers run in parallel per answer, weighted into a hybrid score:

| Layer | Model | Weight | Signal |
|-------|-------|--------|--------|
| 1 | `cross-encoder/nli-deberta-v3-base` | 0.30 | NLI entailment between answer and chunks |
| 2 | RAGAS `faithfulness` (via `gpt-4.1-mini`) | 0.30 | Claim-level factual precision |
| 3 | `cross-encoder/ms-marco-MiniLM-L6-v2` | 0.40 | Answer-context relevance ranking |

`score >= 0.6` → `grounded`, `>= 0.35` → `partially_grounded`, else
`ungrounded` (`GRV_THRESHOLD` in `.env`). If the answer is a refusal
("I could not find relevant information..."), GRV instead runs the
abstention-verification path (§4) to check whether the refusal was
warranted or a missed retrieval.

## Known limitations

- **Retrieval is vector-only, not hybrid.** `retriever.py` uses
  `semantic_hybrid_search_with_score`, but true BM25+vector hybrid search is
  blocked by a `k` keyword-argument conflict in `langchain-community`'s
  `AzureSearch.hybrid_search()`. Tracked as a documented limitation rather
  than fixed.
- **RAGAS Layer 2 is not fully independent** of the model under test — it
  also uses `gpt-4.1-mini` as its judge LLM. Alternatives (Phi-4-mini,
  Llama 3.2 3B via Ollama, Mistral-small) were tried and rejected due to
  rate limits, JSON-schema incompatibility, or deployment capacity on a
  student Azure subscription.
