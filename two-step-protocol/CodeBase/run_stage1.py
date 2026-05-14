import os
import json
import requests
import pandas as pd
from logger import get_logger
from config import load_config
from scientific_fact_checker import ScientificFactChecker

logger = get_logger(__name__)
CONFIG = load_config()

CHAPTERS_DIR = "Chapters"
OUT_DIR = "output"


def check_tavily_connectivity(tavily_key: str) -> bool:
    """
    Pre-flight check: verify Tavily API is reachable and the key is valid
    before starting a potentially long processing run.

    Tries the v2 Bearer-header format first, then falls back to v1 body format.
    Logs a clear warning if Tavily is unavailable so the user knows citations
    will be missing — rather than discovering this after the full run completes.

    Returns True if Tavily is reachable, False otherwise.
    """
    if not tavily_key:
        logger.warning("TAVILY_API_KEY not set — citations will be empty for all rows.")
        return False

    test_query = "cancer biology"
    try:
        # Try v2 Bearer header
        r = requests.post(
            "https://api.tavily.com/search",
            json={"query": test_query, "search_depth": "basic", "max_results": 1},
            headers={"Authorization": f"Bearer {tavily_key}", "Content-Type": "application/json"},
            timeout=15,
        )
        if r.status_code == 200:
            logger.info("Tavily API: pre-flight check PASSED (v2 Bearer auth).")
            return True

        # Try v1 body format
        r2 = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": tavily_key, "query": test_query, "search_depth": "basic"},
            timeout=15,
        )
        if r2.status_code == 200:
            logger.info("Tavily API: pre-flight check PASSED (v1 body auth).")
            return True

        logger.warning(
            f"Tavily API pre-flight FAILED — HTTP {r.status_code}: {r.text[:200]}. "
            "Citations will be empty. Check your TAVILY_API_KEY and quota."
        )
        return False

    except requests.exceptions.Timeout:
        logger.warning("Tavily API pre-flight TIMEOUT. Citations may be empty.")
        return False
    except Exception as e:
        logger.warning(f"Tavily API pre-flight ERROR: {e}. Citations may be empty.")
        return False


def choose_chapter():
    if not os.path.exists(CHAPTERS_DIR):
        os.makedirs(CHAPTERS_DIR)
        print(f"Put your files in '{CHAPTERS_DIR}' and restart.")
        exit()

    files = sorted([f for f in os.listdir(CHAPTERS_DIR) if f.endswith(('.md', '.txt'))])

    if not files:
        print("No files found.")
        exit()

    for i, f in enumerate(files, 1):
        print(f"[{i}] {f}")

    while True:
        user_input = input("Select Chapter Number or Name: ").strip()

        # Case 1: Number input
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(files):
                return os.path.join(CHAPTERS_DIR, files[idx])
            else:
                print("Invalid number. Try again.")
                continue

        # Case 2: Exact filename match
        if user_input in files:
            return os.path.join(CHAPTERS_DIR, user_input)

        # Case 3: Partial match
        matches = [f for f in files if user_input.lower() in f.lower()]
        if len(matches) == 1:
            return os.path.join(CHAPTERS_DIR, matches[0])
        elif len(matches) > 1:
            print("Multiple matches found. Please be more specific:")
            for m in matches:
                print(f"- {m}")
            continue

        print("Invalid input. Enter a number or part of the filename.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    chapter_path = choose_chapter()
    cfg = load_config()

    # Initialization will now check for .env keys automatically
    checker = ScientificFactChecker(cfg)

    # --- Tavily pre-flight check ---
    # Warn immediately if citations will be missing — don't wait until post-run inspection.
    from dotenv import load_dotenv
    load_dotenv()
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    tavily_ok = check_tavily_connectivity(tavily_key)
    if not tavily_ok:
        print(
            "\n⚠️  WARNING: Tavily API is unavailable. Citations will be EMPTY for all rows.\n"
            "   Check your TAVILY_API_KEY in .env and your API quota before proceeding.\n"
            "   Press Enter to continue without citations, or Ctrl+C to abort.\n"
        )
        input()

    logger.info(f"Extracting: {os.path.basename(chapter_path)}")
    text = checker.read_chapter(chapter_path)
    raw = checker.extract_assertions_multi_run(text, os.path.basename(chapter_path))
    master = checker.create_master_list(raw)

    logger.info(f"Auditing {len(master)} assertions...")
    results = checker.run_stage1_factcheck(master)

    # flatten citations for easier manual review in CSV
    for r in results:
        cits = r.get("citations", [])
        if not isinstance(cits, list):
            cits = []
        cleaned = [c for c in cits if isinstance(c, str) and c.strip()]
        r["citations_str"] = " | ".join(cleaned)
        r["citations_count"] = len(cleaned)

    # All unverified items go to Stage 2
    flagged = [r for r in results if r.get("final_verdict") == "Flagged for Review"]

    # Final Saves
    pd.DataFrame(results).to_csv(f"{OUT_DIR}/stage1_results.csv", index=False)

    with open(f"{OUT_DIR}/stage1_flagged_assertions.json", "w", encoding="utf-8") as f:
        json.dump(flagged, f, indent=2, ensure_ascii=False)

    # full Stage 1 results as JSON (keeps citations as a real list)
    with open(f"{OUT_DIR}/stage1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Citation success rate report ---
    total = len(results)
    with_cits = sum(
        1 for r in results
        if isinstance(r.get("citations", []), list) and len(r.get("citations", [])) > 0
    )
    without_cits = total - with_cits
    pct = round(100 * with_cits / total, 1) if total else 0

    logger.info(
        f"Stage 1 Finished. {len(flagged)} items routed to Stage 2. "
        f"Citations: {with_cits}/{total} rows ({pct}%) | "
        f"{without_cits} rows missing citations."
    )

    if without_cits > 0:
        logger.warning(
            f"{without_cits} rows have no citations — Tavily may have failed mid-run. "
            "Check logs above for 'Tavily API error' warnings to diagnose."
        )
    else:
        logger.info("All rows have citations. Tavily working correctly throughout.")


if __name__ == "__main__":
    main()