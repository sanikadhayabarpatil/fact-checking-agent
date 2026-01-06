#!/usr/bin/env python3
"""
Stage 1 – Scenario 1
Robust Gemini + Tavily pipeline (google-genai compatible)
"""

import os, json, re, time
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from google import genai

from logger import logger
from prompt import EXTRACTION_PROMPT, FACT_CHECKING_PROMPT
from rag import SimpleRAGSystem


# -------------------- helpers --------------------

def sleep():
    time.sleep(1.2)

def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower().strip())

def extract_json_array(text: str) -> List[Dict]:
    """
    Robust JSON recovery:
    - strips ``` fences
    - extracts first [...] block
    - json.loads or hard-fail
    """
    if not text:
        return []

    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"\[[\s\S]*\]", cleaned)
    if not m:
        raise ValueError("No JSON array found in Gemini output")

    return json.loads(m.group(0))


def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


# -------------------- Tavily --------------------

class TavilyClient:
    def __init__(self, key: str, domains: List[str]):
        self.key = key
        self.domains = [d.strip().lower() for d in domains if d.strip()]
        self.endpoint = "https://api.tavily.com/search"

    def search(self, query: str, k: int = 4) -> List[Dict]:
        try:
            r = requests.post(
                self.endpoint,
                json={
                    "api_key": self.key,
                    "query": query,
                    "max_results": k,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=30,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            logger.error(f"Tavily error: {e}")
            return []

        if not self.domains:
            return results[:k]

        filtered = []
        for x in results:
            d = domain(x.get("url", ""))
            if any(d == a or d.endswith("." + a) for a in self.domains):
                filtered.append(x)

        return filtered[:k]


# -------------------- Main class --------------------

class ScientificFactChecker:
    def __init__(self, cfg: Dict[str, Any]):
        load_dotenv()

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        self.model = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
        self.client = genai.Client(api_key=key)

        tav_key = os.getenv("TAVILY_API_KEY")
        if not tav_key:
            raise RuntimeError("Missing TAVILY_API_KEY")

        domains = os.getenv("SEARCH_DOMAINS", "").split(",")
        self.tavily = TavilyClient(tav_key, domains)

        self.rag = SimpleRAGSystem(chunk_size=int(cfg.get("CHUNK_SIZE", 500)))
        logger.info(f"ScientificFactChecker ready | model={self.model}")

    # --------------------

    def read_chapter(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def build_kb(self, text: str):
        self.rag.build_index_from_blocks([{
            "text": text,
            "source_title": "Chapter",
            "source_url": ""
        }])

    # -------------------- extraction --------------------

    def extract_assertions_three_runs(self, chapter_path: str) -> List[Dict]:
        text = self.read_chapter(chapter_path)
        prompt = EXTRACTION_PROMPT.format(
            chapter_name=os.path.basename(chapter_path),
            content=text,
        )

        all_items = []

        for i in range(3):
            logger.info(f"Gemini extraction run {i+1}/3")
            r = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            parsed = extract_json_array(r.text)

            for a in parsed:
                if a.get("original_statement"):
                    a.setdefault("optimized_assertion", a["original_statement"])
                    all_items.append(a)

            sleep()

        return all_items

    # -------------------- dedup --------------------

    def deduplicate(self, items: List[Dict]) -> List[Dict]:
        seen = {}
        for a in items:
            k = normalize(a["original_statement"])
            if k not in seen:
                seen[k] = a
        return list(seen.values())

    # -------------------- tavily --------------------

    def attach_tavily(self, items: List[Dict]) -> List[Dict]:
        for i, a in enumerate(items, 1):
            q = a.get("optimized_assertion") or a["original_statement"]
            res = self.tavily.search(q)
            a["index"] = i
            a["tavily_results"] = res
            a["tavily_urls"] = [x.get("url") for x in res if x.get("url")]
        return items

    # -------------------- fact check (batched) --------------------

    def run_stage1(self, items: List[Dict], batch: int = 10) -> List[Dict]:
        out = []

        for i in range(0, len(items), batch):
            chunk = items[i:i+batch]
            payload = []

            for a in chunk:
                payload.append({
                    "index": a["index"],
                    "original_statement": a["original_statement"],
                    "optimized_assertion": a["optimized_assertion"],
                    "RELEVANT_DOCS_LIST": a.get("tavily_results", []),
                    "KNOWLEDGE_BASE_EXCERPTS": self.rag.retrieve_relevant_chunks_with_meta(
                        a["optimized_assertion"], top_k=3
                    ),
                })

            full_prompt = FACT_CHECKING_PROMPT + "\n" + json.dumps(payload)
            r = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
            )

            parsed = extract_json_array(r.text)
            if not parsed:
                raise RuntimeError("Gemini returned empty / invalid JSON")

            out.extend(parsed)
            sleep()

        return out

    # --------------------

    @staticmethod
    def collect_flagged(results: List[Dict]) -> List[Dict]:
        return [r for r in results if r.get("final_verdict") in ("Incorrect", "Flagged for Review")]
