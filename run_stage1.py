#!/usr/bin/env python3

"""
Stage 1 – Run the complete fact-checking pipeline 5 times.
Each run generates:
    output/fact_checking_results_runX.csv
    output/detailed_fact_checking_results_runX.json
without overwriting earlier runs.
"""

from scientific_fact_checker import ScientificFactChecker, load_config
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def run_single_iteration(run_id: int):
    """Runs the full pipeline once and saves results with run_id."""
    
    logger.info(f"\n===== STARTING FACT-CHECKING RUN {run_id} =====")

    # Load config + initialize
    config = load_config()
    checker = ScientificFactChecker(config)

    # ---- Step 1: Extract assertions ----
    #chapter_path = "./Chapters/Chapter 02_ Introduction to Cancer_ A Disease of Deregulation.md"
    chapter_path = "./Chapters/Chapter 11_ Cancer Metabolism_ The Warburg Effect and Beyond.md"
    assertions = checker.extract_assertions(chapter_path)

    if not assertions:
        logger.error(f"Run {run_id}: No assertions extracted.")
        return

    with open(f"assertions_list_run{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(assertions, f, indent=2, ensure_ascii=False)

    # ---- Step 2: Relevant documents ----
    documents_data = checker.find_relevant_documents(assertions)

    with open(f"relevant_documents_run{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(documents_data, f, indent=2, ensure_ascii=False)

    # ---- Step 3: Build KB ----
    knowledge_base = checker.build_knowledge_base(documents_data)

    with open(f"knowledge_base_run{run_id}.txt", "w", encoding="utf-8") as f:
        f.write(knowledge_base)

    # ---- Step 4: Fact-check ----
    results = checker.fact_check_statements(documents_data, knowledge_base)

    # ---- Step 5: Save results with run_id ----
    checker.save_results(results, run_id=run_id)

    logger.info(f"===== COMPLETED FACT-CHECKING RUN {run_id} =====\n")

def main():
    """Runs the whole pipeline 5 times (Stage 1 requirement)."""

    for i in range(1, 6):
        run_single_iteration(i)

if __name__ == "__main__":
    main()
