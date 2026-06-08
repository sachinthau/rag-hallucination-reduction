import uuid
import datetime
from azure.data.tables import TableServiceClient
from src.config.settings import settings


def log_result(result: dict):
    """Logs a pipeline result to Azure Table Storage. Fails silently to avoid breaking the pipeline."""
    try:
        service = TableServiceClient.from_connection_string(settings.STORAGE_CONNECTION_STRING)
        table = service.get_table_client(settings.TABLE_NAME)
        entity = {
            "PartitionKey": result.get("config", "unknown"),
            "RowKey": str(uuid.uuid4()),
            "logged_at": datetime.datetime.utcnow().isoformat(),
            "question": str(result.get("question", ""))[:500],
            "answer": str(result.get("answer", ""))[:1000],
            "latency_ms": int(result.get("latency_ms", 0)),
            "grv_score": str(result.get("grv_score", "")),
            "grv_label": str(result.get("grv_label", "")),
            "in_corpus": str(result.get("in_corpus", "")),
            "question_id": str(result.get("question_id", "")),
            "flagged": str(result.get("flagged", ""))
        }
        table.upsert_entity(entity)
    except Exception as e:
        print(f"[Logger warning] Could not write to Table Storage: {e}")
