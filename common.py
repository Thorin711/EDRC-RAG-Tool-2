# -*- coding: utf-8 -*-
"""
Shared constants, secret/client helpers, and vector-store loaders used by all
four Streamlit apps in this project (app.py, uploader.py, admin.py,
fixer_app.py).

Kept as a flat module at the repo root (not a package) so the existing
Hugging Face Spaces sync workflow (.github/workflows/sync.yml), which force-
pushes this repo as-is, picks it up without any changes.
"""

import os
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Used only if QDRANT_URL isn't set via secrets/env -- kept as a fallback so
# existing deployments that haven't added the secret yet don't break.
DEFAULT_QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io"

COLLECTION_FULL = "full_papers"
COLLECTION_JOURNAL = "journal_papers"
COLLECTION_EDRC = "edrc_papers"
ALL_COLLECTIONS = [COLLECTION_FULL, COLLECTION_JOURNAL, COLLECTION_EDRC]

DB_OPTIONS = {
    "Full Database": COLLECTION_FULL,
    "Journal Articles Only": COLLECTION_JOURNAL,
    "EDRC Only": COLLECTION_EDRC,
}

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
REPORT_SHEET_NAME = "RAG Data Reports"

MODEL_COSTS = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def get_secret(key, default=None):
    """
    Retrieves a secret from environment variables (Hugging Face / Docker) or
    Streamlit secrets (local .streamlit/secrets.toml / Streamlit Cloud), in
    that order. Returns `default` (None unless given) if it's set nowhere.
    """
    # 1. Try Environment Variable (Hugging Face / Docker)
    if key in os.environ:
        return os.environ[key]

    # 2. Try Streamlit Secrets (Local .toml / Streamlit Cloud)
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return default


def get_qdrant_url():
    """Returns the configured Qdrant Cloud URL, falling back to the known default."""
    return get_secret("QDRANT_URL", DEFAULT_QDRANT_URL)


@st.cache_resource
def load_embedding_model():
    """Loads and caches the sentence embedding model from Hugging Face."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_store(_embeddings, collection_name, _url, _api_key):
    """
    Loads and caches a Qdrant-backed vector store for the given collection.
    Replaces the previous load_full_store/load_journal_store/load_edrc_store
    trio that existed (with near-identical bodies) in every app -- Streamlit's
    cache is keyed on `collection_name`, so each collection still gets its own
    cached store.
    """
    client = QdrantClient(url=_url, api_key=_api_key, prefer_grpc=False)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=_embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )
