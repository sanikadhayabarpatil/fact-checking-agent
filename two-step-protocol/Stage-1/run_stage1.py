#!/usr/bin/env python3

import os, json
from datetime import datetime
from config import load_config
from scientific_fact_checker import ScientificFactChecker
from logger import logger

CHAPTERS = "Chapters"
OUT = "output"


def choose_chapter():
    files = sorted(f for f in os.listdir(CHAPTERS) if f.endswith((".md", ".txt")))
    for i, f in enumerate(files, 1):
        print(f"[{i}] {f}")
    while True:
        c = input("Enter chapter number or filename: ").strip()
        if c.isdigit() and 1 <= int(c) <= len(files):
            return os.path.join(CHAPTERS, files[int(c)-1])
        if c in files:
            return os.path.join(CHAPTERS, c)


def dump(p, o):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, ensure_ascii=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    chapter = choose_chapter()

    checker = ScientificFactChecker(load_config())
    text = checker.read_chapter(chapter)
    checker.build_kb(text)

    extracted = checker.extract_assertions_three_runs(chapter)
    master = checker.deduplicate(extracted)
    master = checker.attach_tavily(master)

    results = checker.run_stage1(master)
    flagged = checker.collect_flagged(results)

    dump(f"{OUT}/stage1_master_assertions.json", master)
    dump(f"{OUT}/stage1_initial_results.json", results)
    dump(f"{OUT}/stage1_flagged_assertions.json", flagged)
    dump(f"{OUT}/stage1_metadata.json", {
        "stage": 1,
        "scenario": 1,
        "chapter": os.path.basename(chapter),
        "counts": {
            "master": len(master),
            "results": len(results),
            "flagged": len(flagged),
        },
        "timestamp": datetime.utcnow().isoformat()
    })

    logger.info("Stage 1 complete")
    print(f"Flagged assertions: {len(flagged)}")


if __name__ == "__main__":
    main()
