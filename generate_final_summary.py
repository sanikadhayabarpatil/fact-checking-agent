#!/usr/bin/env python3

"""
Stage 3 – Final summary and publication-ready outputs.

Input:
    output/stage2_focused_verification_results.csv

Outputs:
    output/final_summary_statistics.json
    output/final_results_table.csv
    output/final_verdict_breakdown.txt
    (optional) output/final_confidence_plot.png
"""

import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def main():

    logger.info("\n===== STAGE 3: GENERATING FINAL PUBLICATION SUMMARY =====\n")

    path = OUTPUT_DIR / "stage2_focused_verification_results.csv"
    if not path.exists():
        logger.error("Missing stage2_focused_verification_results.csv")
        return

    df = pd.read_csv(path)

    # ----------------------------
    # Compute global statistics
    # ----------------------------
    total = len(df)
    true_count = (df["Final_Verdict_Stage2"] == "True").sum()
    false_count = (df["Final_Verdict_Stage2"] == "False").sum()

    # Confidence distribution
    df["conf_bucket"] = pd.cut(
        df["Confidence (%)"],
        bins=[0, 60, 80, 99.9, 100.1],
        labels=["Low (≤60%)", "Weak/Medium (60–80%)", "Strong (80–99%)", "Unanimous (100%)"],
        include_lowest=True,
    )

    conf_counts = df["conf_bucket"].value_counts().to_dict()

    # Agreement category counts
    agreement_counts = df["Confidence_Label"].value_counts().to_dict()

    # ----------------------------
    # Save JSON summary
    # ----------------------------
    summary = {
        "total_assertions": total,
        "true_count": int(true_count),
        "false_count": int(false_count),
        "percent_true": round(true_count / total * 100, 2),
        "percent_false": round(false_count / total * 100, 2),
        "confidence_distribution": conf_counts,
        "agreement_levels": agreement_counts,
    }

    json_path = OUTPUT_DIR / "final_summary_statistics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved JSON summary: {json_path}")

    # ----------------------------
    # Save clean table for publication
    # ----------------------------
    pub_cols = [
        "Statement",
        "Assertion",
        "Final_Verdict_Stage2",
        "Confidence (%)",
        "Confidence_Label",
        "Merged_Citations",
    ]
    pub_table = df[pub_cols]
    table_path = OUTPUT_DIR / "final_results_table.csv"
    pub_table.to_csv(table_path, index=False)
    logger.info(f"Saved publication table: {table_path}")

    # ----------------------------
    # Write text summary for manuscript
    # ----------------------------
    text_lines = [
        "===== FACT-CHECKING SUMMARY =====\n",
        f"Total Assertions Verified: {total}",
        f"True Assertions: {true_count} ({round(true_count/total*100, 2)}%)",
        f"False Assertions: {false_count} ({round(false_count/total*100, 2)}%)\n",
        "Confidence Breakdown:",
    ]
    for k, v in conf_counts.items():
        text_lines.append(f"  - {k}: {v}")

    text_lines.append("\nAgreement Levels:")
    for k, v in agreement_counts.items():
        text_lines.append(f"  - {k}: {v}")

    txt_path = OUTPUT_DIR / "final_verdict_breakdown.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(text_lines))

    logger.info(f"Saved readable summary: {txt_path}")

    # ----------------------------
    # (OPTIONAL) Plot confidence histogram
    # ----------------------------
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        df["Confidence (%)"].hist(bins=10)
        plt.title("Confidence Score Distribution")
        plt.xlabel("Confidence (%)")
        plt.ylabel("Frequency")

        plot_path = OUTPUT_DIR / "final_confidence_plot.png"
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Saved confidence histogram: {plot_path}")

    except Exception as e:
        logger.warning(f"Plotting skipped: {e}")

    logger.info("\n===== STAGE 3 COMPLETE =====\n")


if __name__ == "__main__":
    main()
