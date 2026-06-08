# RAG Hallucination Reduction Research Project

MSc Advanced Software Engineering Dissertation  
K.G. Sachintha Udara | University of Westminster via IIT

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.template .env
# Edit .env with your Azure credentials
```

## Run Order

```bash
# 1. Ingest documents
python -m src.ingestion.run_ingestion

# 2. Start the API
uvicorn main:app --reload

# 3. Test with dev set (10 questions)
python -m pytest tests/ -v

# 4. Full evaluation run
python -m src.evaluation.runner
```

## Project Structure

- `src/config/` - Settings and environment loading
- `src/ingestion/` - Document chunking and indexing
- `src/pipeline/` - Three pipeline configurations (A, B, C)
- `src/validator/` - GRV with three layers (cross-encoder, RAGAS, Phi-4)
- `src/evaluation/` - Evaluation runner and metrics
- `data/questions/` - QA dataset (dev_test.json, qa_dataset.json)
- `logs/` - Experiment output CSVs
