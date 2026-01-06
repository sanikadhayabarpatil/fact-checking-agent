"""
config.py

Configuration loader for the Two-Step Fact-Checking Protocol.
Currently supports Stage 1 (Scenario 1).
"""

from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """
    Returns configuration values used across the pipeline.

    Keep this lightweight.
    Do NOT put secrets here — use environment variables instead.
    """
    return {
        # Gemini model used for both extraction and fact-checking
        "MODEL_NAME": "gemini-2.0-flash",

        # Chunk size for chapter-only RAG
        "CHUNK_SIZE": 500,
    }
