#!/usr/bin/env python3

"""
Stage 2 – Step 5:
Focused verification for each flagged assertion.

Input:
    output/master_flagged_assertions.csv

Process:
    For EACH assertion:
        - Run enhanced fact-checking 5 times
        - Convert each verdict to True/False
        - Compute majority vote
        - Compute confidence score

Output:
    output/stage2_focused_verification_results.csv
"""

import pandas as pd
from pathlib import Path
import logging
from scientific_fact_checker import ScientificFactChecker, load_config

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
N_ITER = 5  # 5 independent verification runs


# ----------------------------
# Helpers
# ----------------------------
def verdict_to_bool(v: str) -> bool:
    """
    Stage 2 rule:
        "Correct" → True
        ANYTHING ELSE → False
    """
    return v.strip().lower() == "correct"


def confidence_label(conf: float) -> str:
    if conf == 100:
        return "Unanimous agreement (5/5)"
    elif conf >= 80:
        return "Strong agreement (4/5)"
    elif conf >= 60:
        return "Weak agreement (3/5)"
    else:
        return "Low agreement (≤ 2/5)"


# ----------------------------
# Main Script
# ----------------------------
def main():
    logger.info("\n===== STARTING STAGE 2 – FOCUSED VERIFICATION =====\n")

    master_path = OUTPUT_DIR / "master_flagged_assertions.csv"
    if not master_path.exists():
        logger.error("Missing master_flagged_assertions.csv. Run Stage 1 first.")
        return

    master = pd.read_csv(master_path)

    # Load config + initialize checker ONLY ONCE for speed
    config = load_config()
    checker = ScientificFactChecker(config)

    rows_out = []

    # Process each assertion
    for idx, row in master.iterrows():
        statement = row.get("Statement", "")
        assertion = row.get("Assertion", "")

        logger.info(f"\n--- Verifying assertion {idx + 1}/{len(master)} ---")
        logger.info(f"Statement: {statement[:120]}...")

        iter_labels = []
        iter_bools = []
        iter_citations = []

        # ---- RUN 5 INDEPENDENT VERIFICATIONS ----
        for i in range(N_ITER):
            logger.info(f"  Iteration {i+1}/5 ...")

            result = checker.verify_single_assertion(row, enhanced=True)

            v_label = result["verdict"]
            v_bool = verdict_to_bool(v_label)

            iter_labels.append(v_label)
            iter_bools.append(v_bool)
            iter_citations.append(result.get("citations", []))

        # ---- MAJORITY VOTE ----
        true_count = sum(iter_bools)
        final_bool = true_count >= 3
        final_verdict = "True" if final_bool else "False"

        # ---- CONFIDENCE SCORE ----
        agree_count = sum(1 for b in iter_bools if b == final_bool)
        confidence = (agree_count / N_ITER) * 100
        conf_label = confidence_label(confidence)

        # ---- MERGE CITATIONS (UNION WITHOUT DUPS) ----
        merged_citations = sorted({c for sublist in iter_citations for c in sublist})

        # ---- BUILD OUTPUT ROW ----
        output_row = {
            "Statement": statement,
            "Assertion": assertion,

            # Per-run verdicts
            "Iter1_Verdict": iter_labels[0],
            "Iter2_Verdict": iter_labels[1],
            "Iter3_Verdict": iter_labels[2],
            "Iter4_Verdict": iter_labels[3],
            "Iter5_Verdict": iter_labels[4],

            # Per-run boolean (True/False)
            "Iter1_TrueFalse": iter_bools[0],
            "Iter2_TrueFalse": iter_bools[1],
            "Iter3_TrueFalse": iter_bools[2],
            "Iter4_TrueFalse": iter_bools[3],
            "Iter5_TrueFalse": iter_bools[4],

            # Final results
            "Final_Verdict_Stage2": final_verdict,
            "Confidence (%)": confidence,
            "Confidence_Label": conf_label,

            # Evidence
            "Merged_Citations": "; ".join(merged_citations),
        }

        rows_out.append(output_row)

    # ---- SAVE OUTPUT ----
    df_out = pd.DataFrame(rows_out)
    out_path = OUTPUT_DIR / "stage2_focused_verification_results.csv"
    df_out.to_csv(out_path, index=False)

    logger.info("\n===== STAGE 2 COMPLETE =====")
    logger.info(f"Saved results to: {out_path}\n")


if __name__ == "__main__":
    main()
