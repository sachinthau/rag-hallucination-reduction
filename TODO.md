# Project TODO List

## Known Issues / Tech Debt

- [ ] **Hybrid search bug**: retriever.py currently uses `similarity_search` instead of hybrid search
      due to a `k` argument conflict in langchain-community AzureSearch.
      Root cause: `hybrid_search()` gets multiple values for keyword argument `k`
      Fix: Upgrade langchain-community or wait for patch, then revert retriever.py to:
      `search_type="hybrid"` with semantic ranking enabled
      Impact: Currently using vector-only search instead of hybrid (BM25 + vector).
      Note for dissertation: Document this as a known limitation in Chapter 5.

## Completed
- [x] Azure resources provisioned (AI Foundry, AI Search, Blob Storage)
- [x] Three models deployed (gpt-4.1-mini, text-embedding-3-large, Phi-4)
- [x] Document corpus ingested (2955 chunks, 160 documents)
- [x] Config A tested and working
- [x] Table Storage logger fixed and working

## Next Steps
- [ ] Fix Config B retriever and test
- [ ] Test Config C with GRV
- [ ] Run dev_test.json across all three configs
- [ ] Build qa_dataset.json (100 in-corpus + 50 out-of-corpus questions)
- [ ] Run full evaluation (150 questions x 3 configs)
- [ ] Manual annotation of 50 responses per config
- [ ] Calculate precision, recall, F1, Cohen's Kappa


## Known Warnings (Non-Critical)

- [ ] **Hugging Face unauthenticated warning**: cross-encoder/nli-deberta-v3-base downloads
      anonymously from HF Hub. Warning appears on first download only, model is now cached.
      Fix: Create free HF account, get token, add HF_TOKEN to .env and layer_crossencoder.py
      Impact: None, model works fine without token. Only affects download speed.

## Layer 3 Final Model: cross-encoder/ms-marco-MiniLM-L6-v2
- Selected from Hugging Face Text Ranking category
- 75.6M downloads, most proven reranking model available
- Runs locally, free, no API dependency
- Pair order: (chunk, answer) because ms-marco was trained document-first
- Score normalised using sigmoid function to get 0-1 range
- Layer 1 (DeBERTa): logical entailment signal
- Layer 2 (RAGAS): claim-level factual precision
- Layer 3 (MiniLM): overall relevance ranking signal