# prompt.py
"""
Prompt definitions for Stage 1 extraction and fact checking.

GUARANTEES:
- Safe for Python .format()
- No circular imports
- Backward-compatible symbol names
- No AttributeError / KeyError

v2 CHANGE — ASSERTION_EXTRACTION_PROMPT:
    Replaced exhaustive sentence-level extraction with targeted extraction.
    Previous approach extracted ~1 assertion per sentence (~726 per chapter),
    including definitions, consensus biology, and frameworks — none of which
    are meaningfully fact-checkable. This burned Tavily and Gemini credits
    on claims with no checkable right or wrong answer.

    New approach extracts only two categories of genuinely falsifiable claims:
    CATEGORY 1 — Verification Required: specific numbers, dates, drug approvals,
        study results, guideline recommendations, historical attributions.
    CATEGORY 2 — Expert Sign-Off Required: mechanistic claims that could be
        subtly wrong, contested frameworks, oversimplifications.

    Expected output: ~7-15 assertions per chapter (vs ~726 previously).
    This reduces API cost by ~90% with no loss of useful fact-checking coverage.
"""

# -------------------------------------------------
# Assertion Extraction Prompt — TARGETED (v2)
# -------------------------------------------------

ASSERTION_EXTRACTION_PROMPT = """
You are a precise scientific claim extractor for a cancer biology and therapeutics textbook.

Your job is to extract ONLY genuinely falsifiable claims — claims where something specific
could be factually wrong. You will miss the point if you extract definitions, consensus
biology, or framework descriptions.

A human expert read a chapter of ~8,000 words and found 7 falsifiable claims in 4 minutes.
Your output should be in that range — typically 7 to 20 claims per chapter. Never more than 30.

═══════════════════════════════════════════════════════════════
EXTRACT — CATEGORY 1: VERIFICATION REQUIRED
Claims that can be checked against a named database, paper, or regulatory source.
═══════════════════════════════════════════════════════════════

Include a claim if it contains ANY of the following:

1. A specific number, percentage, rate, or count presented as a current or historical fact
   Examples: survival rates, incidence figures, mortality figures, attributable fractions,
   prevalence estimates, metastasis rates, trial outcomes
   ✓ "HBV causes about 52% of hepatocellular carcinomas"
   ✓ "a 19% decline in new cancer cases and 29% decline in cancer deaths"
   ✓ "lifetime maximum dose of 550 mg/m² for doxorubicin"

2. A drug name with a specific approval, indication, dose limit, or approval date
   ✓ "Cisplatin is FDA approved for the treatment of..."
   ✓ "Cladribine for hairy cell leukemia"
   ✓ "FDA approval of about 100 targeted therapies"

3. A named screening recommendation or clinical guideline with specific thresholds
   ✓ "USPSTF recommends initiating screening at 45 years of age"
   ✓ "mammography screening recommended from age 40"

4. A named study, trial, or publication with a specific finding attributed to it
   ✓ "Hanahan and Weinberg (2000) described six hallmarks of cancer"
   ✓ "Otto Warburg observed aerobic glycolysis in cancer cells in 1924"
   ✓ "the NLST trial could prevent about 8,100 deaths per year"

5. A historical fact with a specific date, named person, or institutional attribution
   ✓ "Rudolf Virchow (1821–1902) founded Zellularpathologie"
   ✓ "The National Cancer Act was signed by Nixon in 1971"
   ✓ "NCI established in 1937"
   ✓ "Nobel Prize awarded to Bishop and Varmus in 1989"

6. A claim about what a named organisation (FDA, WHO, USPSTF, NCI, NCCN) currently
   recommends, requires, or has approved

═══════════════════════════════════════════════════════════════
EXTRACT — CATEGORY 2: EXPERT SIGN-OFF REQUIRED
Claims that require a domain expert — not a database lookup.
═══════════════════════════════════════════════════════════════

Include a claim if it meets ANY of the following:

1. A mechanistic or biological claim more specific than broad textbook consensus
   that could be subtly wrong, incomplete, or contested
   ✓ "cancer cells produce lactic acid exclusively from glucose"
   ✓ "mitochondria are non-functional in cancer cells"

2. A causal claim linking a molecular mechanism to a clinical outcome
   ✓ "P-glycoprotein overexpression is the primary driver of multidrug resistance"

3. A claim presenting a scientific framework as settled when it has been revised
   or is actively debated
   ✓ description of Warburg effect that omits the 2008+ revision
   ✓ hallmarks count that does not reflect the 2022 update

4. Terminology used incorrectly or a passage that is internally inconsistent

5. A claim about what is "currently understood" or "now known" in a fast-moving field
   (immunotherapy, targeted therapy, cancer metabolism, liquid biopsy)

6. A clinical claim about prognosis or treatment that a practising oncologist might dispute

═══════════════════════════════════════════════════════════════
DO NOT EXTRACT — SKIP THESE ENTIRELY
═══════════════════════════════════════════════════════════════

✗ Definitions: "Cancer is a disease of uncontrolled cell growth"
✗ Broad consensus biology: "cancer cells resist apoptosis", "DNA replication errors occur"
✗ Framework descriptions being described as frameworks:
  "The hallmarks of cancer include...", "The Warburg effect is..."
✗ Vague temporal claims: "recently", "in recent years", "thirty years ago"
✗ Mechanistic descriptions with no specific quantitative or attributable claim
✗ Summary or transition sentences
✗ Anything so broadly accepted that no reasonable expert would dispute it

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Return a JSON LIST ONLY. No preamble, no explanation, no markdown.

[
  {{
    "original_statement": "verbatim sentence or phrase from the text — do not paraphrase",
    "optimized_assertion": "self-contained, unambiguous assertion — add subject if pronouns make it unclear",
    "category": "1" or "2",
    "why_check": "one sentence: what specifically could be wrong, outdated, or disputed"
  }}
]

If you find fewer than 7 claims, that is fine — do not pad with low-quality entries.
If you find more than 30, you are extracting too broadly — reapply the SKIP rules.

TEXT:
{content}
"""

# Backward compatibility alias — all existing code imports EXTRACTION_PROMPT
EXTRACTION_PROMPT = ASSERTION_EXTRACTION_PROMPT


# -------------------------------------------------
# Fact Checking Prompt (unchanged)
# -------------------------------------------------

FACT_CHECK_PROMPT = """
You are a scientific fact-checking assistant.

ASSERTION:
{assertion}

EVIDENCE:
{evidence}

Your task is to classify the assertion as ONE of:
- Correct
- Incorrect
- Flagged for Review

DEFINITIONS:
- Correct: Fully supported by authoritative scientific evidence with no unresolved ambiguity.
- Incorrect: Explicitly contradicted by authoritative scientific evidence.
- Flagged for Review: Ambiguous, context-dependent, incomplete, or requires expert judgment.

CRITICAL SAFETY RULE — ABSOLUTE LANGUAGE VETO:

If the assertion contains ANY absolute or universal language such as:
- "all"
- "always"
- "only"
- "exclusively"
- "universally"

THEN:
- It MUST NOT be labeled "Correct"
- Label it "Flagged for Review" unless universality is explicitly proven.

NOTE: The word "first" is NOT grounds for automatic flagging.
Claims like "first-line treatment", "first described by", or historical firsts
are verifiable and should be evaluated on evidence merit alone.

CALIBRATION RULE FOR APPROXIMATE CLAIMS:

If an assertion is:
- Directionally correct AND
- Widely accepted in authoritative scientific literature AND
- Uses approximate or non-absolute language (e.g., "about", "~", "approximately", ranges)

THEN:
- Mark it as "Correct"
- EVEN IF exact numerical values vary across studies

CITATION RULE:
- You MUST populate the "citations" array with URLs from the EVIDENCE section that support your verdict.
- If marking Correct, citations are REQUIRED. Use URLs provided in the evidence above.
- If no supporting URL exists in the evidence, mark as "Flagged for Review" instead.

Return STRICT JSON only:
{{
  "final_verdict": "Correct | Incorrect | Flagged for Review",
  "reasoning": "...",
  "citations": ["<URL from evidence>", ...]
}}
"""

# Backward compatibility alias
FACT_CHECKING_PROMPT = FACT_CHECK_PROMPT


# -------------------------------------------------
# Explicit export guard
# -------------------------------------------------

__all__ = [
    "ASSERTION_EXTRACTION_PROMPT",
    "EXTRACTION_PROMPT",
    "FACT_CHECK_PROMPT",
    "FACT_CHECKING_PROMPT",
]