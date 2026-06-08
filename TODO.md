cat > TODO.md << 'EOF'
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
EOF