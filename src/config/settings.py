from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    GPT4O_DEPLOYMENT: str = os.getenv("AZURE_GPT4O_DEPLOYMENT", "gpt-4.1-mini-deployment")
    EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large-deployment")

    # Azure AI Search
    SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-documents-index")

    # Azure Storage
    STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    BLOB_CONTAINER: str = os.getenv("AZURE_BLOB_CONTAINER", "documents")
    TABLE_NAME: str = os.getenv("AZURE_TABLE_NAME", "experimentlogs")

    # Pipeline parameters
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "5"))
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    LLM_TEMPERATURE: float = 0.0

     # GRV weights and thresholds
    GRV_THRESHOLD: float = float(os.getenv("GRV_THRESHOLD", "0.6"))
    CROSSENCODER_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    GRV_WEIGHT_CROSSENCODER: float = 0.30
    GRV_WEIGHT_RAGAS: float = 0.30
    GRV_WEIGHT_PHI4: float = 0.40

    # Ollama local model for RAGAS Layer 2
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

settings = Settings()
