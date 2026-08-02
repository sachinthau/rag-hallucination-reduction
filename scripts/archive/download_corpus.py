# scripts/download_corpus.py
import requests, os, time
 
DOCS_BASE = "https://raw.githubusercontent.com/MicrosoftDocs/azure-docs/main/articles"
 
# Selected sections relevant to your study
SECTIONS = [
    "ai-services/openai",
    "search",
    "storage/blobs",
    "storage/tables",
    "virtual-network",
]
 
OUTPUT_DIR = "data/corpus"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
def try_download(url: str, dest: str):
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200 and len(resp.text) > 500:
        with open(dest, "w", encoding="utf-8") as f:
            f.write(resp.text)
        return True
    return False
 
# Alternatively: clone the Azure docs repo locally
# git clone --depth=1 https://github.com/MicrosoftDocs/azure-docs.git
# Then copy relevant .md files from articles/ into data/corpus/
