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

## 4. Prepare the QA test dataset (ground truth)

Before anything can be calibrated or evaluated you need a labelled set of
test questions with known-correct answers grounded in the corpus. This is
`data/questions/qa_dataset.json` (200 questions) — it already exists in the
repo, but the process that built it matters for reproducibility if you ever
swap in a different corpus, so it's documented here even though the one-off
scripts that ran it have since been archived out of the repo (see `git log
-- data/questions/` — they were deleted post-use, same cleanup pass that
removed the batch runner).

**1. Automatic drafting.** Questions were drafted in topic batches (7 batches
covering `azure-functions/` and `container-apps/`, plus one batch of
deliberately out-of-corpus questions) by having an LLM read each corpus
document and produce question / expected-answer / source-passage triples.
This is fast but prone to two failure modes: paraphrasing the source text
loosely enough to drift from what it actually says, or outright fabricating
a plausible-sounding answer the doc never states.

**2. Automatic verification, two passes**, to catch those failure modes
before trusting a question:
   - **Pass 1 — passage matching.** For every in-corpus question, fuzzy-match
     `source_passage` against the actual text of `source_doc` (0.75 similarity
     threshold, sliding-window `SequenceMatcher`). Anything below threshold
     gets flagged for review.
   - **Pass 2 — factual-anchor matching.** Whole-passage fuzzy matching alone
     over-flags legitimately correct questions whose passage was paraphrased
     rather than quoted. So only the *flagged* questions get re-checked by
     extracting "anchors" — inline code spans, numbers with units, or the
     `expected_answer` itself — and confirming at least one anchor appears
     literally somewhere in the corpus. A paraphrased-but-correct passage
     still contains the same key facts; a fabricated one doesn't.
   - **Pass 3 — hollow-page cross-check.** Some corpus pages pull their real
     content from an `includes/` file that isn't part of the corpus, leaving
     the page itself as near-empty scaffolding. Any question sourced from one
     of these known-hollow pages is unreliable regardless of its fuzzy-match
     score, so they're cross-referenced and flagged separately.

**3. Manual validation — required, not optional.** Everything flagged as
likely-fabricated, needing manual check, or hollow-page-sourced was reviewed
by hand against the actual corpus document and corrected or dropped. The
automatic passes only catch wording mismatches and missing facts — they
can't confirm a passage is the *right* justification for an answer, only
that it's plausibly present in the text. Every question in the final
`qa_dataset.json` has been through this human check.

If you replace the corpus, repeat the same three-stage process (draft →
verify → manually validate) before trusting a new dataset for calibration or
evaluation — skipping the manual step is how a fabricated "ground truth"
question would silently corrupt every downstream metric.

### Dataset format

Each entry in `qa_dataset.json`:

```json
{
  "id": "Q001",
  "question": "What do you need to add to an existing Azure Function to connect it to other Azure services using bindings?",
  "expected_answer": "You need to add specific binding definitions in your function code or configuration file",
  "source_doc": "azure-functions/add-bindings-existing-function.md",
  "source_passage": "If you want to connect your function to other services by using input or output bindings, you have to add specific binding definitions in your function",
  "in_corpus": true,
  "category": "azure_functions_bindings"
}
```

| Field | Type | Why it's there |
|-------|------|-----------------|
| `id` | string, `Q001`–`Q200` | Stable key used to join results across files: `calculate_metrics.py` merges results CSVs with `data/annotation_template.csv` on this, and `src/utils/logger.py` records it per Table Storage row. |
| `question` | string | The literal prompt sent to `config_a/b/c.query()`. |
| `expected_answer` | string or `null` | The ground-truth answer text; the reference a human annotator checks Config C's actual answer against when filling in `human_label`. `null` for out-of-corpus questions — there's no correct in-corpus answer to give. |
| `source_doc` | string or `null` | Which corpus file the answer must come from; used during dataset verification (above) and for `category_analysis()`'s `in_corpus` grouping. `null` for out-of-corpus questions. |
| `source_passage` | string or `null` | The literal corpus text that supports `expected_answer` — the evidence span the automatic verification passes checked before the question was trusted. `null` for out-of-corpus questions. |
| `in_corpus` | boolean | Drives two different things: which questions feed abstention-threshold calibration (Section 5, must be answerable) and which questions should trigger a refusal instead (`in_corpus: false`, used to test the model doesn't hallucinate when it has nothing to retrieve). |
| `category` | string | Topic/sub-topic tag (e.g. `azure_functions_bindings`) `calculate_metrics.py`'s `category_analysis()` uses to break hallucination rate down by topic in the dissertation results. |

`data/questions/dev_test.json` is a 10-question smoke-test subset with the
same `id` / `question` / `in_corpus` / `expected_answer` / `category` fields
(no `source_doc`/`source_passage`, since it's for quick manual sanity checks
rather than automated verification or metrics).

---

## 5. Calibrate the abstention threshold (true value from the current corpus)

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
this step any time the corpus, index, or QA dataset changes** — `tau` is
only valid for the corpus it was calibrated against.

---

## 6. Try it — compare all three configs on demo questions

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

## 7. Run the API

Run this **from the project root** (same directory as `main.py`), not from
`scripts/` — `main.py` imports `src.pipeline.*` as an absolute package path,
and `.env` is loaded relative to wherever the process starts, so both only
resolve correctly from the root:

```bash
uvicorn main:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I create an Azure Function trigger?", "config": "C"}'
```

`config` is `"A"`, `"B"`, or `"C"` (defaults to `"A"` if omitted).

**What hitting `config: C` here actually does:** it runs the full RAG +
GRV pipeline and returns the real answer, grounding score, and label in the
HTTP response — it's not a no-op. But each call also writes one row to
`AZURE_TABLE_NAME` via `src/utils/logger.py`, and that Table Storage log is
the *only* place ad-hoc API calls are recorded. Hitting `/query` here does
**not** append to `results/results_config_C.csv` — those are produced by a
separate batch process run directly against `qa_dataset.json` (Section 9),
not through this API. Use the API for interactive/manual testing of a single
question; use the batch process for anything that needs to show up in the
evaluation metrics.

---

## 8. Tests

```bash
python -m pytest tests/ -v
```

`tests/test_pipeline.py` exercises all three configs end-to-end (live Azure
calls) plus two unit checks on `src/validator/grv.py` using hand-written
grounded/hallucinated chunk pairs — useful as a fast sanity check that
credentials and deployments are all correct before a full evaluation run.

---

## 9. Full evaluation and metrics

`results/results_config_{A,B,C}.csv` hold the dissertation's evaluation
results — each config run over the full `qa_dataset.json`. There's no bundled
batch-runner script (the original one was a one-off and was removed after
producing these results); to regenerate, loop over
`data/questions/qa_dataset.json` calling `config_a.query()` /
`config_b.query()` / `config_c.query()` per question and collect the
returned dicts into a CSV — that's exactly what each config's `query()`
function returns already.

### 9.1 Human annotation

GRV's reliability is judged by how well its `grv_label` agrees with a human
reading the same answer. That comparison needs a human-labelled CSV first.

```bash
python scripts/generate_annotation_template.py
```

Reads `results/results_config_C.csv` and writes
`data/annotation_template.csv` with one row per question:
`question_id, question, expected_answer, answer, grv_label, human_label, notes`
— `human_label` and `notes` start blank. **It refuses to run if
`data/annotation_template.csv` already exists** (pass `--force` to
overwrite) — the copy already in this repo is the completed dissertation
annotation set (200 rows, hand-labelled), so don't regenerate over it unless
you're re-annotating a fresh Config C run.

**Then, by hand:** open `data/annotation_template.csv` and fill in
`human_label` for every row with `grounded`, `partially_grounded`, or
`ungrounded` — your own judgement of whether Config C's `answer` is actually
supported by `expected_answer` / the source corpus, read independently of
what `grv_label` already says. This is the step nothing can automate: it's
the ground truth GRV is being measured against, so it has to reflect an
honest human read of each answer, not a rubber-stamp of GRV's own label.

### 9.2 Compute Cohen's Kappa

Once `human_label` is filled in for every row you want to include:

```bash
cd scripts
python calculate_metrics.py       # hallucination rate per config, Cohen's Kappa vs human labels,
                                   # per-category breakdown, RAGAS summary -> ../logs/metrics_summary.json
python compute_final_stats.py     # latency aggregation + per-layer GRV score stats
python show_kappa_calculation.py  # step-by-step worked Cohen's Kappa calculation (dissertation appendix)
```

`calculate_metrics.py` and `compute_final_stats.py` use relative paths and
must be run **from inside `scripts/`**.

**How the Kappa is actually calculated** (`show_kappa_calculation.py`,
mirrored inside `calculate_metrics.py`'s `grv_performance()`):

1. Merge `results_config_C.csv` and the annotation CSV on `question_id`.
2. Collapse both `grv_label` and `human_label` to binary:
   `0 = grounded`, `1 = hallucinated` (i.e. `ungrounded` OR
   `partially_grounded` both count as 1 — GRV's job is to catch anything
   that isn't fully grounded, so the fine-grained 3-class label is too
   strict a comparison for agreement scoring).
3. Build the confusion matrix (human rows × GRV columns) → `TN, FP, FN, TP`.
4. `Po` = observed agreement = `(TP + TN) / total`.
5. `Pe` = agreement expected by chance = `P(human=0)·P(grv=0) + P(human=1)·P(grv=1)`.
6. `Kappa = (Po - Pe) / (1 - Pe)` — agreement corrected for how often the
   two raters would match by luck alone. Verified against
   `sklearn.metrics.cohen_kappa_score` in the same script.
7. `Precision`, `Recall`, `F1` are computed on the same binary labels,
   treating `human_label` as truth and `grv_label` as the prediction.

Interpretation scale (Cohen, 1960): `<0.20` slight, `0.21–0.40` fair,
`0.41–0.60` moderate, `0.61–0.80` substantial, `0.81–1.00` almost perfect.
`GRV_THRESHOLD` (`.env`, default `0.6`) is treated as the substantial-agreement
bar the validator needs to clear.

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
  generate_annotation_template.py Build a blank human-annotation CSV from results_config_C.csv
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
abstention-verification path (Section 5) to check whether the refusal was
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
- **Cohen's Kappa is single-annotator.** `human_label` in
  `data/annotation_template.csv` reflects one person's judgement, with no
  second rater to compute inter-annotator agreement against. The 0.72 Kappa
  therefore shows GRV agrees with *that one reviewer*, not with human
  consensus — a second independent annotator would be needed to rule out
  individual labelling bias.
- **Abstention threshold calibration used a modest sample.** `tau` (Section
  5) is the 10th percentile of max-relevance scores over ~140 in-corpus
  questions. That's enough to get a working cut-off, but a small calibration
  set makes the percentile sensitive to individual outlier scores; a larger
  in-corpus question set would produce a more stable, precise `tau` and is
  the first thing to scale up if the corpus or dataset grows.
