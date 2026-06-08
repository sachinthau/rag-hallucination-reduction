# src/utils/logger.py
import uuid
import datetime
from azure.data.tables import TableServiceClient
from src.config.settings import settings


def get_table_client():
    service = TableServiceClient.from_connection_string(settings.STORAGE_CONNECTION_STRING)
    # Create table if it does not exist
    try:
        service.create_table(settings.TABLE_NAME)
        print(f"[Logger] Created table: {settings.TABLE_NAME}")
    except Exception:
        pass  # Table already exists, that is fine
    return service.get_table_client(settings.TABLE_NAME)


def log_result(result: dict):
    try:
        table = get_table_client()
        entity = {
            "PartitionKey": result.get("config", "unknown"),
            "RowKey": str(uuid.uuid4()),
            "logged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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