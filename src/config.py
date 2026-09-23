"""
Centralized config for GraphMind-X.
Everything reads from .env — no hardcoded secrets or paths anywhere else.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphmindpass")

# --- EPO Open Patent Services (OPS) ---
# Free, official, INPADOC-backed — includes Indian patent bibliographic data.
# IP India has no public API of its own, so this is the practical free route.
EPO_OPS_CONSUMER_KEY = os.getenv("EPO_OPS_CONSUMER_KEY", "")
EPO_OPS_CONSUMER_SECRET = os.getenv("EPO_OPS_CONSUMER_SECRET", "")
EPO_OPS_AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_OPS_SEARCH_BIBLIO_URL = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"

# --- Ingestion ---
EPO_COUNTRY_CODE = os.getenv("EPO_COUNTRY_CODE", "IN")
PATENT_SEARCH_KEYWORD = os.getenv("PATENT_SEARCH_KEYWORD", "solid-state battery")
INGESTION_SAMPLE_SIZE = int(os.getenv("INGESTION_SAMPLE_SIZE", "200"))

# --- Ollama (local LLM, zero cost) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

# --- Agent behavior ---
CYPHER_RESULT_LIMIT = int(os.getenv("CYPHER_RESULT_LIMIT", "100"))

# --- spaCy ---
SPACY_MODEL = "en_core_web_sm"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
