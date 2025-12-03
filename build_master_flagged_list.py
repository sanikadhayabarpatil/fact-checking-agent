#!/usr/bin/env python3

"""
Stage 1 — Step 3:
Build a master list of flagged assertions from the 5 runs.

Inputs:
    output/fact_checking_results_run1.csv
    output/fact_checking_results_run2.csv
    ...
    output/fact_checking_results_run5.csv

Output:
    output/master_flagged_assertions.csv
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
N_RUNS = 5

def normalize_text(text: str) -> str:
    """Normalize for deduplication (lowercase, remove extra spaces)."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())

def main():
    logger.info("\n===== BUILDING MASTER FLAGGED ASSERTIONS LIST =====\n")

    dfs = []

    # Load all 5 run results
    for i in range(1, N_RUNS + 1):
        path = OUTPUT_DIR / f"fact_checking_results_run{i}.csv"
        if not path.exists():
            logger.error(f"Missing file: {path}")
            continue
        df = pd.read_csv(path)
        df["run_id"] = i
        dfs.append(df)

    if not dfs:
        logger.error("No run files found. Exiting.")
        return

    all_runs = pd.concat(dfs, ignore_index=True)

    # Filter Incorrect + Flagged
    mask = all_runs["Final Verdict"].isin(["Incorrect", "Flagged for Review"])
    flagged = all_runs[mask].copy()

    logger.info(f"Found {len(flagged)} flagged rows across all runs.")

    # Deduplication key based on Statement + Assertion
    flagged["dedup_key"] = (
        flagged["Statement"].apply(normalize_text)
        + " || "
        + flagged["Assertion"].apply(normalize_text)
    )

    # Deduplicate (keep first occurrence)
    master = (
        flagged
        .sort_values(by=["dedup_key", "run_id"])  # stable ordering
        .drop_duplicates(subset=["dedup_key"])
        .drop(columns=["dedup_key"])
    )

    # Save master file
    output_path = OUTPUT_DIR / "master_flagged_assertions.csv"
    master.to_csv(output_path, index=False)

    logger.info(
        f"\nMaster flagged assertions saved to:\n  {output_path}"
        f"\nTotal unique flagged assertions: {len(master)}\n"
    )

if __name__ == "__main__":
    main()
