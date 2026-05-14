# config.py
"""
Single source of truth for pipeline configuration.

This file exports BOTH:
- CONFIG + load_config() (what your run_stage1 expects)
- GEMINI_MODEL_NAME (what other modules may import)
"""

CONFIG = {
    "MODEL_NAME": "gemini-2.5-flash",
    "CHUNK_SIZE":       5000,
    "CHUNK_OVERLAP":    1000,
    "EXTRACTION_RUNS":  1,      # was 3 — 3x fewer extraction calls; 1 pass is sufficient
    "TEMPERATURE":      0.0,
    "WAIT_TIME":        4,      # was 2 — 4s between fact-check calls (~15/min max)
    "EXTRACTION_WAIT":  6,      # was 4 — 6s between extraction calls (~10/min max)
    "BACKOFF_FACTOR":   60,     # was 20 — waits 60,120,180,240,300s on rate limit
    "GEMINI_TIMEOUT":   120,     # timeout per Gemini call before retry
}

# Backward-compatible constant export (prevents ImportError)
GEMINI_MODEL_NAME = CONFIG["MODEL_NAME"]


def load_config():
    """Returns the configuration dictionary for the agent."""
    return CONFIG


__all__ = ["CONFIG", "load_config", "GEMINI_MODEL_NAME"]