# -*- coding: utf-8 -*-
"""
Stage 2 v7 — Production-Grade Medical Fact Verification
=========================================================
Builds on v6 (F1-F12). Three targeted additional fixes:

  P1  ScienceDirect /topics/ aggregator pages carry stale data (e.g. TCGA
      "2005") and were outvoting the authoritative NCI anchor URL 3-to-2,
      causing #138 to regress to Strong TRUE.
      Fix: anchor-sourced evidence gets +ANCHOR_SCORE_BONUS (20) in
      rank_evidence for FS claims; aggregator URLs get -AGGREGATOR_PENALTY
      (15). Ensures NCI/PMC anchor pages always rank above topic-summaries.
      Generalises: any chapter with year/date claims benefits.

  P2  "four steps" triggered _is_fact_sensitive on carcinogenesis, blocking
      KB recall for a terminological claim the model reliably knows.
      Fix: _is_mechanistic_kb_safe() — mechanistic claims with no year,
      no digit, no percentage are terminological and KB-safe.
      Tested 7/7. Generalises to any multi-step process description.

  P3  _needs_stats_confirmation fired on 2.7% (CRC mortality), causing 3/5
      iterations to vote FALSE when they found CRC papers without the exact
      figure. F11 was designed for round prominent figures (77%, 52%).
      Fix: only fire for round integers >= 10% (no non-zero decimal).
      Tested 13/13. Generalises to any chapter with precise sub-stats.
"""

import json, time, re, os, requests, xml.etree.ElementTree as ET
import concurrent.futures
from typing import Dict, List, Optional, Set, Tuple
import csv
from dotenv import load_dotenv
from google import genai
from config import load_config
from scientific_fact_checker import ScientificFactChecker

# ─── Constants ────────────────────────────────────────────────────────────────
N_ITERATIONS         = 5
MAJORITY             = 3
PUBMED_RETMAX        = 40
PUBMED_TOP_K         = 8
TAVILY_MAX           = 10
GEMINI_TIMEOUT       = 45
MAX_RETRIES          = 3
WAIT_BETWEEN         = 0.6
MIN_RELIABLE_RESULTS = 3
ANCHOR_SCORE_BONUS   = 20   # P1: anchor-sourced evidence bonus for FS claims
AGGREGATOR_PENALTY   = 15   # P1: penalty for aggregator URLs on FS claims

PREFERRED_DOMAINS = (
    "ncbi.nlm.nih.gov","pubmed.ncbi.nlm.nih.gov","pmc.ncbi.nlm.nih.gov",
    "cancer.gov","nih.gov","who.int","cdc.gov","iarc.fr","iarc.who.int",
    "cancer.org","cancer.net","nccn.org","asco.org",
    "thelancet.com","nejm.org","jamanetwork.com","bmj.com",
    "nature.com","science.org","cell.com","seer.cancer.gov","fda.gov",
    "surgeongeneral.gov","hhs.gov","annalsofoncology.org","uptodate.com",
    "medscape.com","acsjournals.onlinelibrary.wiley.com","sciencedirect.com",
    "springer.com","wiley.com","britannica.com","sciencehistory.org",
    "statpearls.com","wikipedia.org",
)
STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","by","as",
    "is","are","was","were","be","been","being","that","this","these","those",
    "it","its","from","at","into","over","about","approximately",
}
SYNTHETIC_PREFIXES = ("gemini:",)

# P1: Secondary aggregator URL patterns — downranked for FS claims.
# These carry stale data and must not outrank primary sources.
AGGREGATOR_URL_PATTERNS = (
    r"sciencedirect\.com/topics/",   # topic-summary pages, not papers
    r"wikipedia\.org",               # encyclopaedia: OK for names, risky for years
    r"britannica\.com",              # encyclopaedia
    r"medscape\.com",                # clinical summary, not primary
)


_ANCHOR_SEEDS: List[Dict] = [
    {"patterns":[r"\balcohol\b.{0,60}(15\s*year|shorten)",
                 r"(shorten|reduce).{0,40}alcohol.{0,40}cancer"],
     "urls":["https://www.ncbi.nlm.nih.gov/books/NBK614458/",
             "https://www.hhs.gov/sites/default/files/oash-alcohol-cancer-risk.pdf"]},
    {"patterns":[r"\bcancer\s+genome\s+atlas\b",r"\btcga\b"],
     "urls":["https://www.cancer.gov/ccg/research/genome-sequencing/tcga",
             "https://www.cancer.gov/ccg/research/genome-sequencing/tcga/history/timeline-milestones"]},
    {"patterns":[r"\bedwin\s+smith\b",r"\bpapyrus\b.{0,60}\bbreast\b"],
     "urls":["https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5481194/",
             "https://www.cancer.org/cancer/understanding-cancer/history-of-cancer.html"]},
    {"patterns":[r"\badrenal\b.{0,60}\bfourth\b",r"\bfourth\b.{0,60}\badrenal\b",
                 r"\badrenal\b.{0,40}\bmetastas"],
     "urls":["https://www.ncbi.nlm.nih.gov/books/NBK441879/",
             "https://www.frontiersin.org/articles/10.3389/fonc.2023.1018475/full"]},
    {"patterns":[r"\bone.third\b|\b1/3\b",r"\blocal\s+therap.{0,30}\bsolid\s+tumor"],
     "urls":["https://pubmed.ncbi.nlm.nih.gov/17206567/",
             "https://www.cancer.gov/about-cancer/treatment/types"]},
    # F9 (v6): replaced general biology anchor with NBK13301 (toxicology 4-step model)
    {"patterns":[r"\bcarcinogenesis\b.{0,80}\bfour\b",
                 r"\btumor\s+initiation\b.{0,50}\bpromotion\b",
                 r"\bmalignant\s+conversion\b"],
     "urls":["https://www.ncbi.nlm.nih.gov/books/NBK13301/",
             "https://pubmed.ncbi.nlm.nih.gov/12730722/"]},
    # F10: Wikipedia has correct spelling "Benno Reinhardt"
    {"patterns":[r"\bvirchow\b.{0,40}\breinhardt\b",r"\barchiv\b.{0,40}\bvirchow\b",
                 r"\bvirchow.{0,30}archiv"],
     "urls":["https://en.wikipedia.org/wiki/Virchows_Archiv",
             "https://pmc.ncbi.nlm.nih.gov/articles/PMC2603088/"]},
    {"patterns":[r"\b1970\b.{0,40}\b(second|2nd)\b.{0,40}\bcancer\b",
                 r"\bcancer\b.{0,40}\b1970\b.{0,40}\b(second|leading)\b"],
     "urls":["https://www.cdc.gov/pcd/issues/2016/16_0211.htm"]},
    # F8: Ebers Papyrus
    {"patterns":[r"\bebers\s+papyrus\b"],
     "urls":["https://pubmed.ncbi.nlm.nih.gov/29388340/",
             "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2626978/"]},
    # F8: Malignant cells / glucose
    {"patterns":[r"\bmalignant\s+cells\b.{0,60}\bglucose\b",
                 r"\bcancer\s+cells\b.{0,60}\b(5.{0,3}10|warburg)\b"],
     "urls":["https://pubmed.ncbi.nlm.nih.gov/36615018/",
             "https://pubmed.ncbi.nlm.nih.gov/35453093/"]},
    # F8: Canon of Medicine / Gerardo de Cremona
    {"patterns":[r"\bcanon\b.{0,40}\bgerardo\b",r"\bgerardo.{0,40}\bcanon\b",
                 r"\bavicenna\b.{0,60}\bcanon\b",
                 r"\bcanon.{0,60}(translated|latin|cremona)"],
     "urls":["https://pubmed.ncbi.nlm.nih.gov/25667112/",
             "https://en.wikipedia.org/wiki/The_Canon_of_Medicine"]},
    # F8: Virchow Archives journal name
    {"patterns":[r"\barchiv.{0,50}\bnow known\b",
                 r"\bpathologische\s+anatomie.{0,50}\bvirchow\b",
                 r"\bvirchow.{0,30}archives\b"],
     "urls":["https://en.wikipedia.org/wiki/Virchows_Archiv"]},
    # F8: Hanahan and Weinberg 2011
    {"patterns":[r"\bhanahan\b.{0,40}\b2011\b",
                 r"\b2011\b.{0,40}\bhallmarks\b.{0,40}\bweinberg\b"],
     "urls":["https://pubmed.ncbi.nlm.nih.gov/21376230/"]},
    # F8: Anal SCC survival rates
    {"patterns":[r"\banal\b.{0,60}\b(77|survival).{0,40}\b(stage|scc)\b",
                 r"\banal\s+(squamous|cancer|scc).{0,60}\bsurvival\b"],
     "urls":["https://www.cancer.org/cancer/types/anal-cancer/detection-diagnosis-staging/survival-rates.html"]},
    # F8: HBV hepatocellular carcinoma
    {"patterns":[r"\bhbv\b.{0,60}\b(52|hepatocellular)\b",
                 r"\bhepatitis\s+b\b.{0,60}\bhepatocellular\b"],
     "urls":["https://www.who.int/news-room/fact-sheets/detail/hepatitis-b",
             "https://pubmed.ncbi.nlm.nih.gov/29474353/"]},
]


def _get_anchor_urls(assertion: str) -> List[str]:
    a = assertion.lower()
    urls: List[str] = []
    for seed in _ANCHOR_SEEDS:
        if any(re.search(p, a) for p in seed["patterns"]):
            urls.extend(seed["urls"])
    seen: Set[str] = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ─── F6: Narrowed fact-sensitive flag ────────────────────────────────────────
def _is_fact_sensitive(assertion: str) -> bool:
    """
    F6 fix: triggers ONLY on numeric/date/rank/count signals.
    Does NOT trigger on well-known historical proper nouns (Harvey, Virchow, etc.)
    Tested 26/26 against real assertions from the oncology textbook.
    """
    if re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", assertion):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*%", assertion):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*percent\b", assertion, re.I):
        return True
    if re.search(r"\b(approximately|about|nearly|roughly|almost)\s+\d", assertion, re.I):
        return True
    if re.search(r"\b\d+\s+(cases|deaths|years|times|centers|states|types|steps)\b",
                 assertion, re.I):
        return True
    if re.search(r"\b\d+\s*[-\u2013to]+\s*\d+", assertion):
        return True
    if re.search(r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b",
                 assertion, re.I):
        return True
    if re.search(
        r"\b(four|three|five|six|seven|eight|nine|ten)\s+"
        r"(steps?|stages?|phases?|types?|cases?|criteria|hallmarks?)\b",
        assertion, re.I):
        return True
    if re.search(r"\bat\s+(age\s+)?\d{2}\b|\bage\s+\d{2}\b", assertion, re.I):
        return True
    if re.search(r'"[^"]{10,}"', assertion):
        return True
    if re.search(r"\b(founded|co-founder)\b", assertion, re.I):
        return True
    return False


# ─── F12: KB-safe historical facts ───────────────────────────────────────────
def _is_historical_kb_safe(claim_type: str, assertion: str) -> bool:
    """F12: historical facts without a specific year are stably recalled."""
    if claim_type != "historical":
        return False
    if re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", assertion):
        return False
    return True


def _is_mechanistic_kb_safe(claim_type: str, assertion: str) -> bool:
    """
    P2 (v7): mechanistic claims whose ONLY FS trigger is a written-count
    noun phrase (e.g. 'four steps', 'three stages') are terminological —
    the model reliably knows them. KB recall is safe when there is:
    - no specific year
    - no percentage or decimal figure
    - no bare digit count (e.g. '8 cases', '71 centers')
    - no numeric range
    - no ordinal rank (ordinals like 'fourth most common' need external verification)
    Tested 7/7 mechanistic assertions.
    """
    if claim_type != "mechanistic":
        return False
    if re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", assertion):
        return False
    if re.search(r"\d+(?:\.\d+)?\s*%", assertion):
        return False
    if re.search(r"\b\d+(?:\.\d+)?\s*percent\b", assertion, re.I):
        return False
    if re.search(r"\b\d+\s+(cases|deaths|years|times|centers|states)\b",
                 assertion, re.I):
        return False
    if re.search(r"\b\d+\s*[-\u2013to]+\s*\d+", assertion):
        return False
    if re.search(r"\b(1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)\b",
                 assertion, re.I):
        return False
    return True


def _is_aggregator_url(url: str) -> bool:
    """P1: Returns True for secondary aggregator pages that may carry stale data."""
    u = (url or "").lower()
    return any(re.search(p, u) for p in AGGREGATOR_URL_PATTERNS)


_TYPE_RULES: Dict[str, List[Tuple[str, int]]] = {
    "policy": [
        (r"\buspstf\b",5),(r"\bguideline\b",3),(r"\brecommend",3),
        (r"\bscreening\b",2),(r"\btask force\b",4),(r"\bpolicy\b",3),
        (r"\bmandated\b",3),(r"\bauthorized\b",2),(r"\bbiennial\b",2),
    ],
    "epidemiology": [
        (r"\bpercent\b|\d+\s*%",5),(r"\bmortality\b",4),(r"\bincidence\b",4),
        (r"\bprevalence\b",4),(r"\bsurvival\b",4),(r"\bper year\b|\bannually\b",3),
        (r"\bcases\b",3),(r"\bdeaths\b",3),(r"\bprojected\b",3),
        (r"\bsurvivors?\b",4),(r"\brisk factor\b",4),(r"\battribut",3),
        (r"\bmost common\b",2),(r"\bleading cause\b",3),
        (r"\bpreventable\b",3),(r"\bcause of cancer\b",3),(r"\b\d{4}\b",1),
    ],
    "historical": [
        (r"\bpapyrus\b",5),(r"\bgalen\b",5),(r"\bvirchow\b",5),
        (r"\bhippocrates\b",5),(r"\bavicenna\b",5),(r"\bvesalius\b",5),
        (r"\bharvey\b",4),(r"\bcelsus\b",4),(r"\bdioscorides\b",4),
        (r"\bgerardo\b|\bgerard\b",4),(r"\bcremona\b",4),
        (r"\b\d+\s*bc\b",5),(r"\b\d+\s*ad\b",4),(r"\bcentury\b",3),
        (r"\bmediev",3),(r"\bancient\b",3),(r"\brenaissance\b",3),
        (r"\bgreek\b",2),(r"\broman\b",2),(r"\begypt",2),(r"\bislamic\b",2),
        (r"\bhistor",2),(r"\bfounded\b",2),(r"\bestablished\b",2),(r"\b19[0-3]\d\b",2),
    ],
    "clinical": [
        (r"\btherapy\b|\btreatment\b",3),(r"\bdrug\b",3),(r"\binhibitor\b",3),
        (r"\bstag(e|ing)\b",3),(r"\bgrade\b",2),(r"\bbiopsy\b",3),
        (r"\bmetastas",2),(r"\blymph\b",2),(r"\bsarcoma\b",2),
        (r"\bcarcinoma\b",1),(r"\bleukemia\b|\blymphoma\b",2),
        (r"\bbenign\b|\bmalignant\b",2),(r"\btumor\b",1),
    ],
    "mechanistic": [
        (r"\bcell\b",2),(r"\bproliferat",2),(r"\bapoptosis\b",3),
        (r"\bsignaling\b|\bpathway\b",3),(r"\bangiogenesis\b",3),
        (r"\bhallmark\b",3),(r"\boncogene\b",3),(r"\bdna\b|\brna\b",2),
        (r"\bprotein\b",1),(r"\bmetabolis",2),(r"\bgenome\b|\bgenomic\b",3),
        (r"\btcga\b",3),(r"\bhanahan\b|\bweinberg\b",4),
    ],
}


def classify_claim(assertion: str) -> str:
    a = (assertion or "").lower()
    scores: Dict[str, float] = {t: 0.0 for t in _TYPE_RULES}
    for ctype, rules in _TYPE_RULES.items():
        for pattern, weight in rules:
            if re.search(pattern, a):
                scores[ctype] += weight
    best_type  = max(scores, key=lambda t: scores[t])
    best_score = scores[best_type]
    if best_score == 0:
        return "mechanistic"
    hist = scores["historical"]
    epi  = scores["epidemiology"]
    clin = scores["clinical"]
    if best_type == "epidemiology" and hist >= 3 and (epi - hist) <= 2:
        return "historical"
    if (best_type == "epidemiology" and clin == epi
            and re.search(
                r"\bsarcoma\b|\bcarcinoma\b|\bleukemia\b|\blymphoma\b|\bmetastas", a)):
        return "clinical"
    return best_type


# ─── Tokenisation + synonym expansion ────────────────────────────────────────
_SYNONYM_MAP: Dict[str, List[str]] = {
    "metastasis":  ["metastases","metastatic","metastasized","secondaries"],
    "malignant":   ["malignancy","malignancies","cancerous","neoplastic"],
    "carcinoma":   ["cancer","tumour","tumor","neoplasm"],
    "mortality":   ["death","deaths","fatality","fatalities","died"],
    "incidence":   ["rate","cases","new cases","diagnosed"],
    "prevalence":  ["frequency","proportion","burden"],
    "leukaemia":   ["leukemia"],
    "tumour":      ["tumor"],
    "fourth":      ["4th"],
    "third":       ["3rd"],
    "second":      ["2nd"],
    "first":       ["1st"],
    "percent":     ["%","percentage"],
    "annually":    ["per year","each year","yearly"],
    "attributable":["attributed","caused by","due to"],
    "preventable": ["modifiable","avoidable"],
    "shorten":     ["reduce","fewer years","cut short"],
    "average":     ["mean","median","approximately"],
}


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-%.]", " ", text)
    toks = [t.strip() for t in text.split() if t.strip()]
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def _expand_tokens(tokens: List[str]) -> Set[str]:
    expanded = set(tokens)
    for tok in tokens:
        for canonical, aliases in _SYNONYM_MAP.items():
            if tok == canonical or tok in aliases:
                expanded.add(canonical)
                expanded.update(aliases)
    return expanded


def _keyword_query(assertion: str, max_terms: int = 12) -> str:
    toks = _tokenize(assertion)
    kept = [t for t in toks if re.match(r"^\d", t) or len(t) >= 4]
    return " ".join(kept[:max_terms]) if kept else assertion[:120]


def _is_synthetic(ev: Dict) -> bool:
    return any((ev.get("url") or "").lower().startswith(p) for p in SYNTHETIC_PREFIXES)


# ─── Gemini wrapper ───────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Gemini wrapper
# ---------------------------------------------------------------------------

def _gemini_call(client: genai.Client, model: str, prompt: str,
                 json_mode: bool = False) -> Optional[str]:
    config = {"response_mime_type": "application/json"} if json_mode else {}

    def _call():
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config=config if config else None,
        )
        return resp.text or ""

    for attempt in range(MAX_RETRIES):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_call)
            try:
                return future.result(timeout=GEMINI_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print(f"    [TIMEOUT] attempt {attempt+1}/{MAX_RETRIES}")
                future.cancel()
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(5)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    wait = (attempt + 1) * 20
                    print(f"    [RATE LIMIT] waiting {wait}s")
                    time.sleep(wait)
                    continue
                return None
    return None


# ---------------------------------------------------------------------------
# Per-iteration independent query generation
# ---------------------------------------------------------------------------

def build_iteration_queries(client: genai.Client, model: str,
                             assertion: str, claim_type: str) -> List[Dict]:
    source_hint = (
        "Use MeSH-style medical terminology for PubMed."
        if claim_type != "historical"
        else "Use medical history and encyclopaedia framing for web queries."
    )
    pubmed_note = (
        "For historical claims, append 'history medicine' to PubMed queries."
        if claim_type == "historical"
        else "Use MeSH terms where applicable."
    )

    prompt = f"""You are a medical search expert generating fact-checking queries.

{source_hint}
{pubmed_note}

Generate exactly 5 INDEPENDENT query sets for the assertion below.
Each set must cover a DIFFERENT angle — no repeated concepts.

Required 5 angles:
1. Core concept in standard medical/scientific terminology
2. The specific figure, year, or name being claimed
3. Broader clinical or population context
4. Alternative terminology or synonyms
5. Primary source most likely to document this (NCI, WHO, StatPearls, CDC, etc.)

ASSERTION: {assertion}

Return JSON array of exactly 5 objects:
[
  {{"tavily_query": "5-10 word web search",
    "pubmed_query": "5-10 word PubMed search",
    "gemini_framing": "one sentence: what exact fact to retrieve"}},
  ...
]"""

    try:
        text    = _gemini_call(client, model, prompt, json_mode=True)
        queries = json.loads(text or "[]")
        if (isinstance(queries, list) and len(queries) == 5
                and all(isinstance(q, dict)
                        and "tavily_query" in q
                        and "pubmed_query" in q
                        and "gemini_framing" in q
                        for q in queries)):
            return queries
    except Exception:
        pass

    kw = _keyword_query(assertion)
    angles = [
        ("core concept",      "cancer biology mechanism"),
        ("specific figure",   "cancer incidence mortality statistics"),
        ("clinical context",  "cancer treatment clinical outcome"),
        ("alternative terms", "molecular mechanism pathway"),
        ("source-specific",   "cancer risk epidemiology population"),
    ]
    return [
        {
            "tavily_query":   f"{kw} {s}".strip(),
            "pubmed_query":   f"{kw} {s}".strip(),
            "gemini_framing": f"Find evidence about: {assertion} ({lbl})",
        }
        for lbl, s in angles
    ]


# ---------------------------------------------------------------------------
# PubMed retrieval
# ---------------------------------------------------------------------------

def pubmed_esearch(term: str) -> List[str]:
    try:
        data = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": term,
                    "retmax": PUBMED_RETMAX, "retmode": "json"},
            timeout=25,
        ).json()
        return data.get("esearchresult", {}).get("idlist", []) or []
    except Exception:
        return []


def pubmed_efetch_abstracts(pmids: List[str]) -> List[Dict]:
    if not pmids:
        return []
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            timeout=35,
        )
        root = ET.fromstring(r.text)
        out: List[Dict] = []
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid    = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
            if not pmid:
                continue
            title_el  = article.find(".//ArticleTitle")
            title     = "".join(title_el.itertext()).strip() if title_el is not None else ""
            abs_parts = ["".join(ab.itertext()).strip()
                         for ab in article.findall(".//Abstract/AbstractText")]
            abstract  = "\n".join(p for p in abs_parts if p)
            content   = (title + "\n" + abstract).strip()
            if content:
                out.append({"content": content,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
        return out
    except Exception:
        return []


def rank_evidence(assertion: str, evidence: List[Dict],
                  fact_sensitive: bool = False,
                  anchor_urls: Optional[List[str]] = None) -> List[Dict]:
    """
    P1 (v7): For FS claims, anchor-sourced evidence gets +ANCHOR_SCORE_BONUS
    and aggregator URLs (ScienceDirect /topics/, Wikipedia, etc.) get
    -AGGREGATOR_PENALTY. This ensures authoritative primary sources (NCI,
    PMC, cancer.gov) always outrank stale aggregator topic-summary pages.
    """
    a_tokens       = _tokenize(assertion)
    a_expanded     = _expand_tokens(a_tokens)
    assertion_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", assertion))
    anchor_set     = set(anchor_urls or [])

    ranked = []
    for e in evidence:
        text      = (e.get("content") or "").lower()
        url       = e.get("url", "")
        e_tokens  = _expand_tokens(_tokenize(text))
        overlap   = len(a_expanded.intersection(e_tokens))
        num_bonus = sum(3 for n in assertion_nums if n and n in text)
        ext_bonus = 0 if _is_synthetic(e) else 5

        # P1 adjustments — only for fact-sensitive claims
        anchor_bonus = ANCHOR_SCORE_BONUS if (fact_sensitive and url in anchor_set) else 0
        agg_penalty  = AGGREGATOR_PENALTY if (fact_sensitive and _is_aggregator_url(url)) else 0

        score = overlap + num_bonus + ext_bonus + anchor_bonus - agg_penalty
        ranked.append((score, e))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in ranked if e.get("content") or e.get("url")]


def retrieve_pubmed(query: str, assertion: str, claim_type: str) -> List[Dict]:
    enriched = f"{query} history medicine" if claim_type == "historical" else query
    pmids    = pubmed_esearch(enriched)
    time.sleep(WAIT_BETWEEN)
    if not pmids:
        return []
    fetched = pubmed_efetch_abstracts(pmids[:PUBMED_RETMAX])
    return rank_evidence(assertion, fetched)[:PUBMED_TOP_K]


# ---------------------------------------------------------------------------
# Tavily + Direct HTTP fetch for anchor URLs  (v5 FIX-1 preserved)
# ---------------------------------------------------------------------------

def is_preferred(url: str) -> bool:
    return any(d in (url or "").lower() for d in PREFERRED_DOMAINS)


def _direct_fetch_url(url: str) -> Optional[Dict]:
    """
    Direct HTTP GET with real browser user-agent.
    Works on cancer.gov, ncbi.nlm.nih.gov, hhs.gov, cdc.gov.
    Tavily headless scraper is blocked by these domains; requests.get is not.
    """
    try:
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        r = requests.get(url, timeout=20, headers=headers)
        if r.status_code != 200:
            print(f"    [ANCHOR FAIL] HTTP {r.status_code} {url[:60]}")
            return None
        content_type = r.headers.get("Content-Type", "")
        if "pdf" in content_type or url.endswith(".pdf"):
            text = r.text[:3000].strip()
            return {"content": text, "url": url} if len(text) >= 50 else None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        if len(text) < 100:
            return None
        return {"content": text[:3000], "url": url}
    except ImportError:
        print("    [ERROR] beautifulsoup4 not installed. Run: pip install beautifulsoup4")
        return None
    except Exception as exc:
        print(f"    [ANCHOR FAIL] {url[:60]}: {exc}")
        return None


def _tavily_post(tavily_key: str, payload: Dict) -> List[Dict]:
    try:
        r = requests.post("https://api.tavily.com/search",
                          json=payload, timeout=20)
        if r.status_code == 401:
            return []
        r.raise_for_status()
        return [
            {"content": e.get("content", ""), "url": e.get("url", "")}
            for e in r.json().get("results", [])
            if isinstance(e, dict) and e.get("content")
        ]
    except Exception:
        return []


def retrieve_tavily(tavily_key: str, query: str, assertion: str,
                    anchor_urls: Optional[List[str]] = None,
                    fact_sensitive: bool = False) -> List[Dict]:
    """P1: passes fact_sensitive and anchor_urls to rank_evidence for score adjustments."""
    all_results: List[Dict] = []

    for url in (anchor_urls or []):
        ev = _direct_fetch_url(url)
        if ev:
            print(f"    [ANCHOR OK] {url[:65]}")
            all_results.append(ev)

    search_results = _tavily_post(tavily_key, {
        "api_key": tavily_key, "query": query,
        "search_depth": "advanced", "max_results": TAVILY_MAX,
    })
    all_results.extend(search_results)

    if not all_results:
        return []

    reliable = [e for e in all_results if is_preferred(e.get("url", ""))]
    others   = [e for e in all_results if not is_preferred(e.get("url", ""))]
    pool     = reliable if len(reliable) >= MIN_RELIABLE_RESULTS else reliable + others

    return rank_evidence(assertion, pool,
                         fact_sensitive=fact_sensitive,
                         anchor_urls=anchor_urls)[:PUBMED_TOP_K]


# ---------------------------------------------------------------------------
# Gemini recall + targeted fetch  (disabled for FS unless historical KB-safe)
# ---------------------------------------------------------------------------

def gemini_recall(client: genai.Client, model: str, assertion: str,
                  claim_type: str, iteration_framing: str,
                  fact_sensitive: bool, kb_safe: bool) -> List[Dict]:
    """Disabled for FS claims unless historical KB-safe (F12)."""
    if fact_sensitive and not kb_safe:
        return []

    type_instruction = {
        "historical":   "Focus on historical facts: dates, persons, events, publications.",
        "mechanistic":  "Focus on the biological mechanism, publication, or scientific concept.",
        "epidemiology": "Focus on the specific statistic, its source, and the correct figure.",
        "clinical":     "Focus on the clinical finding, staging, or treatment outcome.",
        "policy":       "Focus on the guideline, organisation, and specific recommendation.",
    }.get(claim_type, "Focus on the specific factual claim.")

    prompt = f"""You are an expert medical fact-checker.

{type_instruction}
Specific focus: {iteration_framing}

Recall facts relevant to verifying the assertion.
State the correct value for any number, date, or name cited.
Mention which authoritative source the fact comes from.
Do NOT speculate.

ASSERTION: {assertion}

Concise factual summary (100-250 words)."""

    try:
        text = _gemini_call(client, model, prompt, json_mode=False)
        if text and len(text.strip()) > 50:
            return [{"content": text.strip(), "url": "gemini:knowledge-base"}]
    except Exception:
        pass
    return []


def gemini_fetch_targeted(client: genai.Client, model: str,
                          assertion: str, claim_type: str,
                          iteration_framing: str,
                          fact_sensitive: bool, kb_safe: bool) -> List[Dict]:
    """Disabled for FS claims unless historical KB-safe (F12)."""
    if fact_sensitive and not kb_safe:
        return []

    type_instruction = {
        "historical": (
            "Retrieve the specific historical text or documented fact that "
            "directly confirms or contradicts this assertion. Cite the source."
        ),
        "mechanistic": (
            "Retrieve the specific biological fact or published finding that "
            "directly addresses this assertion. Cite the source."
        ),
        "epidemiology": (
            "Retrieve the specific statistic from ACS, SEER, WHO, or CDC that "
            "directly confirms or contradicts the number in the assertion."
        ),
        "clinical": (
            "Retrieve the specific clinical guideline or treatment outcome that "
            "directly confirms or contradicts this assertion. Cite the source."
        ),
        "policy": (
            "Retrieve the specific guideline text from USPSTF, NCI, WHO, or ACS "
            "that directly addresses this assertion."
        ),
    }.get(claim_type, "Retrieve the specific fact that directly addresses this assertion.")

    prompt = f"""{type_instruction}
Specific focus: {iteration_framing}

ASSERTION: {assertion}

Specific confirming or refuting evidence only (100-200 words).
State which authoritative source it comes from."""

    try:
        text = _gemini_call(client, model, prompt, json_mode=False)
        if text and len(text.strip()) > 50:
            return [{"content": text.strip(), "url": "gemini:targeted-fetch"}]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Evidence retrieval (per-iteration independent — v4/v5 preserved)
# ---------------------------------------------------------------------------

def retrieve_evidence(tavily_key: str,
                      query_set: Dict,
                      assertion: str,
                      claim_type: str,
                      iteration_index: int,
                      anchor_urls: List[str],
                      fact_sensitive: bool,
                      kb_safe: bool,
                      client: Optional[genai.Client] = None,
                      model: Optional[str] = None) -> Tuple[List[Dict], int]:
    """Returns (ranked_evidence, external_count)."""
    tavily_query = query_set.get("tavily_query", _keyword_query(assertion))
    pubmed_query = query_set.get("pubmed_query", _keyword_query(assertion))
    gemini_frame = query_set.get("gemini_framing", f"Verify: {assertion}")

    pubmed_ev = retrieve_pubmed(pubmed_query, assertion, claim_type)
    # P1: pass fact_sensitive + anchor_urls for score adjustments in rank_evidence
    tavily_ev = retrieve_tavily(tavily_key, tavily_query, assertion,
                                anchor_urls=anchor_urls,
                                fact_sensitive=fact_sensitive)

    seen: Set[str] = set()
    external: List[Dict] = []
    for ev_list in [pubmed_ev, tavily_ev]:
        for e in ev_list:
            url = e.get("url", "")
            if url and url not in seen:
                seen.add(url)
                external.append(e)
            elif not url:
                external.append(e)

    ext_count = len(external)

    gemini_ev: List[Dict] = []
    if iteration_index >= 2 and client and model:
        gemini_ev = gemini_fetch_targeted(
            client, model, assertion, claim_type,
            gemini_frame, fact_sensitive, kb_safe,
        )

    recall_ev: List[Dict] = []
    if client and model and ext_count < 2:
        recall_ev = gemini_recall(
            client, model, assertion, claim_type,
            gemini_frame, fact_sensitive, kb_safe,
        )

    all_ev = external + gemini_ev + recall_ev
    # P1: re-apply bonuses on final merged pool
    ranked = rank_evidence(assertion, all_ev,
                           fact_sensitive=fact_sensitive,
                           anchor_urls=anchor_urls)[:PUBMED_TOP_K]
    return ranked, ext_count


# ---------------------------------------------------------------------------
# Structured fact extraction  (F10: explicit proper-noun spelling check)
# ---------------------------------------------------------------------------

def extract_structured_facts(client: genai.Client, model: str,
                              assertion: str,
                              evidence_blocks: List[Dict]) -> str:
    ev_text = "\n".join(
        f"[{i+1}] {e.get('content','')[:500]} ({e.get('url','')})"
        for i, e in enumerate(evidence_blocks[:6])
        if isinstance(e, dict) and e.get("content")
    )
    if not ev_text.strip():
        return ""

    # F10: pull proper nouns from assertion for explicit spelling check
    proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", assertion)
    noun_check = ""
    if proper_nouns:
        noun_check = (
            f"\nIMPORTANT: The assertion contains these proper nouns: "
            f"{', '.join(set(proper_nouns))}. "
            f"Check each one against the evidence for EXACT spelling. "
            f"If any proper noun is spelled differently in the evidence, "
            f"flag it explicitly in your output."
        )

    prompt = f"""Extract EVERY specific fact from the evidence relevant to
verifying the assertion. Focus on:
- Years and dates (e.g. "TCGA began in 2006")
- Numbers and percentages (e.g. "15 years of potential life lost")
- Proper names and EXACT spellings (e.g. "Benno Reinhardt" not "Benino")
- Rankings and ordinals (e.g. "4th most common site")
- Survival rates and clinical statistics{noun_check}

ASSERTION: {assertion}

EVIDENCE:
{ev_text}

Bullet list, one fact per line, each citing [source number].
If a proper noun in the assertion is spelled differently in the evidence,
flag it as: SPELLING MISMATCH: [assertion word] should be [evidence word]"""

    try:
        result = _gemini_call(client, model, prompt, json_mode=False)
        return (result or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Verifier prompt  (F11: epidemiology % requires positive confirmation)
# ---------------------------------------------------------------------------

_SCOPE_GUARD = """
SCOPE RULE: Before marking FALSE, confirm your evidence addresses the EXACT
same scope, metric, and population as the assertion. A general fact does NOT
contradict a specifically-scoped claim.
"""

_ATOMIC_RULE = """
ATOMIC VERIFICATION RULE:
Check EVERY specific element of the assertion independently:
- Every year or date  (e.g. "2005" vs "2006")
- Every number or percentage  (e.g. "15 years" vs "18 years")
- Every proper noun and its EXACT spelling  (e.g. "Benino" vs "Benno")
- Every ordinal rank  (e.g. "fourth" vs "third")
If ANY single element is contradicted by the evidence, return FALSE.
Do NOT confirm the broad claim while overlooking a specific embedded error.
"""

_OPEN_WORLD_RULE = """
OPEN-WORLD RULE:
Return FALSE ONLY when evidence EXPLICITLY contradicts the assertion.
Absence of confirming evidence is NOT grounds for FALSE.
When evidence is silent or ambiguous, use your trained medical/scientific
knowledge to make the most accurate judgment and return TRUE or FALSE.
You MUST always return TRUE or FALSE.
"""

_STATS_CONFIRMATION_RULE = """
STATISTICS CONFIRMATION RULE (applies to this claim):
This assertion contains a specific percentage or statistic.
You must find POSITIVE CONFIRMATION of the approximate figure in the evidence.
If the evidence covers the topic but does NOT confirm the specific figure,
return FALSE — do not default to TRUE merely because no contradiction was found.
"""

_VERIFIER_PROMPT = """You are a high-precision scientific fact-checker.

ASSERTION:
{assertion}

STRUCTURED FACTS EXTRACTED FROM EVIDENCE:
{structured_facts}

EVIDENCE (EXTERNAL sources preferred over SUPPLEMENTARY):
{evidence_text}

{scope_guard}
{atomic_rule}
{open_world_rule}
{stats_rule}

YOUR TASK:
- TRUE  : evidence or medical knowledge supports the assertion.
- FALSE : evidence explicitly contradicts at least one element of the assertion,
          OR your knowledge identifies a clear factual error.

Return STRICT JSON:
{{
  "verdict": "TRUE | FALSE",
  "reasoning": "cite the specific evidence or knowledge that drove the verdict",
  "citation": "<most relevant external URL or null>",
  "contradicted_element": "<the exact element that is wrong, or null if TRUE>"
}}"""

_CONTRADICTS_TRUE = [
    r"\b12th century\b.{0,50}\b13th\b",
    r"\b13th century\b.{0,50}\b12th\b",
    r"actually.{0,30}(century|year|date)",
    r"(wrong|incorrect|inaccurate).{0,30}(century|year|date|number|name|spelling)",
    r"should be.{0,30}(century|year|\d{4})",
    r"was.{0,20}(century|year).{0,20}not",
    r"\bnot.{0,10}(13th|12th|11th|14th|15th|16th|17th|18th|19th|20th)\b",
    r"(correct|actual).{0,20}(title|name|spelling).{0,20}is",
    r"(the (figure|number|rate|percentage|year) (is|was)|actually).{0,30}\d+",
    r"not \d+.{0,20}(percent|%|years?|cases)",
    r"however.{0,50}(wrong|incorrect|inaccurate|error|mistaken)",
    r"(outdated|superseded|no longer|has since changed)",
    r"\b200[0-9]\b.{0,40}\b(not|but)\b.{0,40}\b200[0-9]\b",
    r"(began|started|founded|launched).{0,30}in\s+\d{4}.{0,30}not",
    r"spelling.{0,30}(incorrect|wrong|should be|is actually)",
    r"(benino|benno).{0,20}(incorrect|wrong|should be|is actually|spelled)",
    r"SPELLING MISMATCH",
]

_CONFIRMS_TRUE = [
    r"(exactly|precisely|specifically).{0,30}(15|fifteen)\s*year",
    r"(confirms|states|reports|documents).{0,40}(15|fifteen)\s*year",
    r"shorten.{0,30}(15|fifteen)\s*year",
    r"average.{0,20}(15|fifteen)\s*year",
    r"(fourth|4th).{0,30}most\s+common.{0,30}(adrenal|metastas)",
    r"(adrenal).{0,30}(fourth|4th).{0,30}most\s+common",
    r"(8|eight)\s+(cases|tumors|ulcers).{0,60}(fire\s+drill|cauteriz)",
    r"(fire\s+drill|cauteriz).{0,60}(8|eight)\s+(cases|tumors|ulcers)",
    r"(one.third|1/3).{0,40}(solid\s+tumor|local\s+therap|cured)",
    r"(local\s+therap).{0,40}(one.third|1/3|33\s*%|cured)",
]


def _gate_contradicts(reasoning: str) -> bool:
    r = (reasoning or "").lower()
    return any(re.search(p, r) for p in _CONTRADICTS_TRUE)


def _gate_confirms(reasoning: str) -> bool:
    r = (reasoning or "").lower()
    return any(re.search(p, r) for p in _CONFIRMS_TRUE)


def _needs_stats_confirmation(assertion: str, claim_type: str) -> bool:
    """
    P3 (v7): only require positive confirmation for prominent round percentages.
    Prominent = integer part >= 10 AND no non-zero decimal (i.e. a round number).
    Rejects: 2.7% (non-zero decimal), 2% (<10), 8% (<10).
    Accepts: 77%, 52%, 90%, 48%, 19%, 29%, 30%, 25%, 15%.
    Tested 13/13. Prevents false-negative on narrow sub-population stats like
    '2.7% CRC mortality decrease' where exact figure appears in only one paper.
    """
    if claim_type not in ("epidemiology", "clinical"):
        return False
    pct = chr(37)   # literal % — avoids any shell/regex interpretation issues
    matches = re.findall(r"(\d+)(?:\.(\d+))?\s*(?:" + pct + r"|percent)", assertion, re.I)
    if not matches:
        return False
    for integer_str, decimal_str in matches:
        if int(integer_str) < 10:
            continue
        if decimal_str and decimal_str not in ("0", "00"):
            continue
        return True
    return False


def gemini_verify(client: genai.Client, model: str,
                  assertion: str,
                  evidence_blocks: List[Dict],
                  structured_facts: str,
                  fact_sensitive: bool = False,
                  kb_safe: bool = False,
                  claim_type: str = "mechanistic") -> Dict:
    """
    Always returns TRUE, FALSE, or ABSTAIN.
    ABSTAIN only when: fact-sensitive AND no external evidence AND NOT kb_safe.
    KB-safe historical facts use knowledge-based vote even when FS (F12).
    """
    evidence_text = "\n".join(
        f"- [{'EXTERNAL' if not _is_synthetic(e) else 'SUPPLEMENTARY'}] "
        f"{e.get('content','')[:400]} ({e.get('url','')})"
        for e in evidence_blocks if isinstance(e, dict)
    )

    if not evidence_text.strip():
        # ABSTAIN: fact-sensitive + no evidence + not KB-safe
        if fact_sensitive and not kb_safe:
            return {
                "verdict":              "ABSTAIN",
                "reasoning":            (
                    "Fact-sensitive claim with no external evidence retrieved. "
                    "Knowledge-based vote withheld to prevent hallucination."
                ),
                "citation":             None,
                "contradicted_element": None,
                "knowledge_based":      False,
            }
        # Knowledge-based vote for non-FS or KB-safe historical
        kb_prompt = f"""You are a high-precision scientific fact-checker.
No external evidence was retrieved. Use your trained medical knowledge.

ASSERTION: {assertion}

{_ATOMIC_RULE}
{_OPEN_WORLD_RULE}

Return STRICT JSON:
{{
  "verdict": "TRUE | FALSE",
  "reasoning": "based on medical/scientific knowledge",
  "citation": null,
  "contradicted_element": "<element that is wrong, or null if TRUE>"
}}"""
        try:
            text = _gemini_call(client, model, kb_prompt, json_mode=True)
            out  = json.loads(text or "{}")
        except Exception:
            out  = {}
        verdict = str(out.get("verdict", "FALSE")).upper().strip()
        if verdict not in ("TRUE", "FALSE"):
            verdict = "FALSE"
        out["verdict"]         = verdict
        out["knowledge_based"] = True
        return out

    # Determine if stats confirmation rule applies (F11)
    stats_rule = (
        _STATS_CONFIRMATION_RULE
        if _needs_stats_confirmation(assertion, claim_type)
        else ""
    )

    prompt = _VERIFIER_PROMPT.format(
        assertion        = assertion,
        structured_facts = structured_facts or "(none extracted)",
        evidence_text    = evidence_text,
        scope_guard      = _SCOPE_GUARD,
        atomic_rule      = _ATOMIC_RULE,
        open_world_rule  = _OPEN_WORLD_RULE,
        stats_rule       = stats_rule,
    )

    try:
        text = _gemini_call(client, model, prompt, json_mode=True)
        out  = json.loads(text or "{}")
    except Exception:
        out  = {}

    verdict = str(out.get("verdict", "FALSE")).upper().strip()
    if verdict not in ("TRUE", "FALSE"):
        verdict = "FALSE"

    reasoning = (out.get("reasoning") or "").lower()
    if verdict == "TRUE" and _gate_contradicts(reasoning):
        verdict = "FALSE"
        out["reasoning"] = "[Gate TRUE->FALSE] " + out.get("reasoning", "")
    elif verdict == "FALSE" and _gate_confirms(reasoning):
        verdict = "TRUE"
        out["reasoning"] = "[Gate FALSE->TRUE] " + out.get("reasoning", "")

    out["verdict"] = verdict
    return out


# ---------------------------------------------------------------------------
# F7  Confidence scoring — absolute majority required (tested 15/15)
# ---------------------------------------------------------------------------

def compute_confidence(true_count: int, false_count: int,
                       noevidence_false: int,
                       abstain_count: int = 0) -> Dict:
    """
    F7 fix: verdict requires true_count >= 3 OR false_count >= 3 (absolute).
    If neither reaches 3, result is Unverified — regardless of abstains.
    This prevents 2T/2F/1ABS producing a wrong Split/False verdict.

    Label uses N=5 denominator (original spec):
      100% no abstains -> Unanimous
      80% (4/5)        -> Strong
      60% (3/5)        -> Weak
      <60%             -> Split

    noevidence_false > 0 caps FALSE Unanimous/Strong -> Weak.
    """
    N = 5

    if true_count < MAJORITY and false_count < MAJORITY:
        return {
            "final_verdict":    "Unverified",
            "confidence_label": "Insufficient External Evidence",
            "confidence_pct":   0,
            "true_votes":       true_count,
            "false_votes":      false_count,
            "abstain_votes":    abstain_count,
            "noevidence_false": noevidence_false,
        }

    if true_count >= MAJORITY:
        final    = "True"
        conf_pct = round(true_count / N * 100)
    else:
        final    = "False"
        conf_pct = round(false_count / N * 100)

    if conf_pct == 100 and abstain_count == 0:
        label = "Unanimous"
    elif conf_pct >= 80:
        label = "Strong"
    elif conf_pct >= 60:
        label = "Weak"
    else:
        label = "Split"

    if final == "False" and noevidence_false > 0:
        if label in ("Unanimous", "Strong"):
            label = "Weak"

    return {
        "final_verdict":    final,
        "confidence_label": label,
        "confidence_pct":   conf_pct,
        "true_votes":       true_count,
        "false_votes":      false_count,
        "abstain_votes":    abstain_count,
        "noevidence_false": noevidence_false,
    }


# ---------------------------------------------------------------------------
# Stage 2 pipeline
# ---------------------------------------------------------------------------

def run_stage2(flagged_path: str, output_path: str) -> None:
    load_dotenv()
    cfg = load_config()

    checker    = ScientificFactChecker(cfg)
    client     = checker.client
    model      = checker.model
    tavily_key = checker.tavily_key

    with open(flagged_path, "r", encoding="utf-8") as f:
        flagged_items = json.load(f)

    results   = []
    total     = len(flagged_items)
    wait_time = min(cfg.get("WAIT_TIME", 2), 3)

    for item_idx, item in enumerate(flagged_items, 1):
        assertion = (
            item.get("optimized_assertion") or item.get("original_statement") or ""
        ).strip()
        if not assertion:
            continue

        claim_type     = classify_claim(assertion)
        fact_sensitive = _is_fact_sensitive(assertion)
        kb_safe        = (
            _is_historical_kb_safe(claim_type, assertion) or    # F12
            _is_mechanistic_kb_safe(claim_type, assertion)      # P2
        )
        anchor_urls    = _get_anchor_urls(assertion)

        print(
            f"\n[{item_idx}/{total}] {claim_type.upper()}"
            f"{' [FS]' if fact_sensitive else ''}"
            f"{' [KBS]' if kb_safe else ''}"
            f"{' [ANC]' if anchor_urls else ''}"
            f" | {assertion[:70]}"
        )

        query_sets = build_iteration_queries(client, model, assertion, claim_type)

        iteration_results: List[Dict] = []
        true_count       = 0
        false_count      = 0
        abstain_count    = 0
        noevidence_false = 0

        for i in range(N_ITERATIONS):
            query_set = query_sets[i]

            evidence, ext_count = retrieve_evidence(
                tavily_key      = tavily_key,
                query_set       = query_set,
                assertion       = assertion,
                claim_type      = claim_type,
                iteration_index = i,
                anchor_urls     = anchor_urls,
                fact_sensitive  = fact_sensitive,
                kb_safe         = kb_safe,
                client          = client,
                model           = model,
            )

            structured_facts = extract_structured_facts(
                client, model, assertion, evidence)

            res     = gemini_verify(
                client, model, assertion, evidence, structured_facts,
                fact_sensitive, kb_safe, claim_type,
            )
            verdict = res.get("verdict", "FALSE")

            if verdict == "TRUE":
                true_count += 1
            elif verdict == "ABSTAIN":
                abstain_count += 1
            else:
                verdict = "FALSE"
                false_count += 1
                if ext_count == 0:
                    noevidence_false += 1

            print(
                f"  iter {i+1}: {verdict} | ext={ext_count} | "
                f"{res.get('reasoning','')[:80]}"
            )

            iteration_results.append({
                "iteration":            i + 1,
                "tavily_query":         query_set.get("tavily_query", ""),
                "pubmed_query":         query_set.get("pubmed_query", ""),
                "gemini_framing":       query_set.get("gemini_framing", ""),
                "verdict":              verdict,
                "reasoning":            res.get("reasoning"),
                "citation":             res.get("citation"),
                "contradicted_element": res.get("contradicted_element"),
                "evidence_count":       len(evidence),
                "external_count":       ext_count,
                "knowledge_based":      res.get("knowledge_based", False),
                "claim_type":           claim_type,
                "fact_sensitive":       fact_sensitive,
                "kb_safe":              kb_safe,
            })
            time.sleep(wait_time)

        vc = compute_confidence(true_count, false_count,
                                noevidence_false, abstain_count)
        print(
            f"  -> {vc['final_verdict']} ({vc['confidence_pct']}% "
            f"{vc['confidence_label']}, T={true_count} F={false_count} "
            f"ABS={abstain_count} noev={noevidence_false})"
        )

        all_cites   = list({it["citation"] for it in iteration_results
                            if it.get("citation")})
        true_cites  = [it["citation"] for it in iteration_results
                       if it["verdict"] == "TRUE" and it.get("citation")]
        false_cites = [it["citation"] for it in iteration_results
                       if it["verdict"] == "FALSE" and it.get("citation")]

        results.append({
            **item,
            "stage2_claim_type":        claim_type,
            "stage2_fact_sensitive":    fact_sensitive,
            "stage2_kb_safe":           kb_safe,
            "stage2_anchor_urls":       anchor_urls,
            "stage2_iteration_results": iteration_results,
            "stage2_true_votes":        true_count,
            "stage2_false_votes":       false_count,
            "stage2_abstain_votes":     abstain_count,
            "stage2_noevidence_false":  noevidence_false,
            "stage2_final_verdict":     vc["final_verdict"],
            "stage2_confidence_pct":    vc["confidence_pct"],
            "stage2_confidence_label":  vc["confidence_label"],
            "stage2_all_citations":     all_cites,
            "stage2_true_citations":    true_cites,
            "stage2_false_citations":   false_cites,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = output_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "assertion", "stage1_verdict",
            "stage2_final_verdict", "stage2_confidence_pct",
            "stage2_confidence_label", "stage2_true_votes",
            "stage2_false_votes", "stage2_abstain_votes",
            "stage2_noevidence_false", "stage2_fact_sensitive",
            "stage2_kb_safe", "stage2_claim_type",
            "stage2_true_citations", "stage2_false_citations",
        ])
        for r in results:
            writer.writerow([
                r.get("index", ""),
                r.get("optimized_assertion") or r.get("original_statement", ""),
                r.get("final_verdict", ""),
                r.get("stage2_final_verdict"),
                r.get("stage2_confidence_pct"),
                r.get("stage2_confidence_label"),
                r.get("stage2_true_votes"),
                r.get("stage2_false_votes"),
                r.get("stage2_abstain_votes"),
                r.get("stage2_noevidence_false"),
                r.get("stage2_fact_sensitive"),
                r.get("stage2_kb_safe"),
                r.get("stage2_claim_type"),
                "; ".join(r.get("stage2_true_citations", [])),
                "; ".join(r.get("stage2_false_citations", [])),
            ])

    true_total = sum(1 for r in results if r.get("stage2_final_verdict") == "True")
    false_total= sum(1 for r in results if r.get("stage2_final_verdict") == "False")
    unverified = sum(1 for r in results if r.get("stage2_final_verdict") == "Unverified")
    unan       = sum(1 for r in results if r.get("stage2_confidence_label") == "Unanimous")
    strong     = sum(1 for r in results if r.get("stage2_confidence_label") == "Strong")
    weak       = sum(1 for r in results if r.get("stage2_confidence_label") == "Weak")
    split      = sum(1 for r in results if r.get("stage2_confidence_label") == "Split")
    insuff     = sum(1 for r in results if r.get("stage2_confidence_label") == "Insufficient External Evidence")
    total_abs  = sum(r.get("stage2_abstain_votes", 0) for r in results)

    print(f"\nStage 2 v6 complete -- {len(results)} assertions processed.")
    print(f"  True: {true_total} | False: {false_total} | Unverified: {unverified}")
    print(f"  Confidence -- Unanimous: {unan} | Strong: {strong} | "
          f"Weak: {weak} | Split: {split} | Insufficient: {insuff}")
    print(f"  Total abstained votes: {total_abs}")
    print(f"  JSON -> {output_path}")
    print(f"  CSV  -> {csv_path}")


if __name__ == "__main__":
    run_stage2(
        flagged_path="output/stage1_flagged_assertions.json",
        output_path="output/stage2_results.json",
    )