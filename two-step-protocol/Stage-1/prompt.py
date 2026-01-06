"""
prompt.py

Prompts for the Two-Step Fact-Checking Protocol
Stage 1 – Initial Screening (High Sensitivity)
"""

# ============================================================
# ASSERTION EXTRACTION PROMPT (UNCHANGED)
# ============================================================

EXTRACTION_PROMPT = """
# SCIENTIFIC ASSERTION EXTRACTION TASK

## YOUR ROLE
You are a scientific fact-checking assistant.

## OBJECTIVE
Extract the MAXIMUM number of clear, factual, testable assertions from the chapter below.
Process EVERY sentence carefully.

## WHAT COUNTS AS AN ASSERTION
INCLUDE:
- Concrete factual claims
- Cause–effect relationships
- Experimental or research findings
- Historical, statistical, or quantitative claims

EXCLUDE:
- Definitions without claims
- Narrative or motivational text
- Opinions or hypotheses

## OUTPUT FORMAT (STRICT)
Return ONLY a valid JSON array.

Each element MUST be exactly:

{{
  "original_statement": "<factual claim>",
  "optimized_assertion": "<search-optimized version>"
}}

## Chapter: {chapter_name}

## Content:
{content}

Return ONLY the JSON array.
"""

# ============================================================
# FACT CHECKING PROMPT (UPDATED – RELIABILITY FIX)
# ============================================================

FACT_CHECKING_PROMPT = """
# SCIENTIFIC FACT-CHECKING TASK
# Stage 1 – Initial Screening (Scenario 1)

## YOUR ROLE
You are a scientific fact-checking agent.

You evaluate assertions using:
1) External evidence (Tavily search results)
2) Scientific reasoning
3) Chapter context (supporting only)

## CRITICAL RELIABILITY RULE (NON-NEGOTIABLE)

An assertion can be marked **"Correct" ONLY IF**:
- At least ONE external citation URL is provided
- AND the citation clearly supports the claim

If **no external citation URL is available**, you MUST NOT mark "Correct",
even if the claim is well-known, plausible, or supported by the chapter.

In that case, choose **"Flagged for Review"**.

## INPUT STRUCTURE
Each item contains:
- index
- original_statement
- optimized_assertion
- RELEVANT_DOCS_LIST (Tavily results)
- KNOWLEDGE_BASE_EXCERPTS

## VERDICT RULES

Choose EXACTLY ONE:

### "Correct"
ONLY IF:
- At least ONE external citation URL is included
- AND the evidence directly supports the claim

### "Incorrect"
ONLY IF:
- External evidence clearly contradicts the claim

### "Flagged for Review"
USE WHENEVER:
- No external citation URL is available
- Evidence is weak, indirect, mixed, or generic
- Claim is numerical, historical, or study-specific
- Verification relies mainly on the chapter text
- You are not highly confident

⚠️ When in doubt, choose "Flagged for Review".

## OUTPUT FORMAT (STRICT)
Return ONLY a valid JSON array.

Each element MUST be:

{{
  "index": <int>,
  "final_verdict": "Correct" | "Incorrect" | "Flagged for Review",
  "reasoning": "<brief justification>",
  "citations": ["<external URL>", ...]
}}

## IMPORTANT
- "Correct" MUST have citations.length ≥ 1
- If citations is empty → verdict CANNOT be "Correct"

## BEGIN INPUT
<ITEMS>
"""
