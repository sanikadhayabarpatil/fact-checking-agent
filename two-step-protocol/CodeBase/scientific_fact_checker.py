# -*- coding: utf-8 -*-
import os
import json
import time
import re
import requests
from typing import Any, Dict, List, Optional
import csv
from google import genai
from dotenv import load_dotenv

import prompt


class ScientificFactChecker:
    """
    Stage-1 Scientific Fact-Checking Agent

    DESIGN INTENT:
    - Clear well-established medical facts quickly using model knowledge
    - Only route genuinely uncertain/unverifiable claims to Stage 2
    - Target: ~30-60 Flagged items out of ~300, not 150+
    - Zero false Incorrects: a correct fact discarded as wrong has no rescue path
    - Minimise false Corrects: wrong facts passed as trusted reach students unchecked

    OVERRIDE PIPELINE (applied in order after model verdict):
    Confidence gate: Correct/Incorrect with confidence < 0.95 -> Flagged for Review
    F. Structural high-risk detector - proactively flags 5 error categories
    A. Contradiction detector  - Correct -> Flagged when reasoning admits an error
    B. Absolute-language veto  - Correct -> Flagged for universal language patterns
    C. Numeric self-check      - Correct -> Flagged via secondary model call on exact stats
    D. Hard-coded false-Correct catches - Correct -> Incorrect for confirmed recurring errors
    E. Hard-coded false-Incorrect rescue - Incorrect -> Correct for confirmed false negatives

    FULL FIX HISTORY:
    Session 1 - Replaced FACT_CHECKING_PROMPT + citation gate with STAGE1_PROMPT.
    Session 2 - Tightened absolute-language veto (removed \\ball\\b, \\bonly\\b, \\bsole\\b).
    Session 3 - Contradiction detector with zero-FP substring signals.
    Session 4 - Century mismatch detector (structured, not substring).
    Session 5 - Numeric check tightened to exact single % and post-1930 years only.
    Session 6 - Override D patterns: Stage 0 not-cancerous, 90% environmental, CRC 74%.
    Session 7 - Expanded KNOWN FACTS prompt block; UV, systemic veins, CRC survival,
                USPSTF 2024, H&W 2000 six hallmarks, Remak+Virchow, Chosroes I.
    Session 8 - Seven targeted fixes validated on 273-row production CSV:
                1. KNOWN FACTS expanded: brain mets, historical descriptions, six hallmarks,
                   TCGA 2006, Virchow 1856, historical medicine context.
                2. CONTRADICTION_SIGNALS: added "two years off", "off by", "slightly off",
                   "outdated", "superseded", "is now outdated", "was updated", etc.
                   Fixed "was not " -> scoped variants to stop F5 false-flag.
                3. Override D: added TCGA 2005 and Virchow 1854 patterns.
                4. Override E (new): rescues I1 (brain mets), I4 (black bile historical),
                   I7 (six hallmarks H&W 2000).
                Simulation result: 3/3 false-Incorrects fixed, 4/4 false-Corrects caught,
                0 unexpected changes to true-correct rows, F5 false-flag eliminated.
    Session 9 - Three reliability fixes + citation pipeline repair (282-row production CSV):
                Issues found in audit: 1 false-Incorrect (row 84), 2 false-Corrects (rows 261, 32).
                1. Override D Pattern 5 tightened: Virchow 1854 now requires 'professor',
                   'became', or 'appointed' in assertion - not 'patholog' which fires on
                   book titles containing 'Pathologie' (row 84 false-Incorrect).
                2. Override D Pattern 6 (NEW): USPSTF mammography 50-74 -> Incorrect.
                   2024 guideline changed to age 40+. Model inconsistently admits or omits
                   this update; hard-code makes it deterministic (row 261 false-Correct).
                3. Override D Pattern 7 (NEW): 1700 BC cancer reference -> Flagged for Review.
                   No confirmed source from 1700 BC. Edwin Smith Papyrus = ~1600 BC.
                   Model's '1700 ~= 1600 BC' logic is backwards (row 32 false-Correct).
                4. CONTRADICTION_SIGNALS expanded: replaced bare "while not " / "though not "
                   with scoped versions to prevent false flags on supporting qualifications.
                   Added: "though the guidelines", "have since been updated", "updated in 20",
                   "guidelines were updated" - catches model admissions of stale guidelines.
                5. search_tavily() rewritten: adds Authorization Bearer header (Tavily v2),
                   falls back to legacy body format on 401, logs all errors with status codes.
                   Root cause of 0/282 citations: silent exception swallowing + likely v1/v2
                   API format mismatch. Now failures are visible in logs.
    Session 10 - Two false-Incorrect rescues + Tavily rate-limit fix (303-row production CSV):
                Issues found in audit:
                  - 2 false-Incorrects (rows 157, 211): no false-Corrects confirmed.
                  - Citations: rows 1-149 have 0 citations, rows 150-303 have full coverage.
                    Root cause: Tavily 429 rate-limit hit on first ~149 requests, no retry logic.
                1. Override E Rescue 4 (NEW): benign + 'usually/typically/generally' +
                   'not damaging/life-threatening' -> Correct.
                   Row 157: model marks Incorrect ignoring 'usually' qualifier. Standard
                   NCI/ACS/textbook language. 'Usually' explicitly covers exceptions.
                2. Override E Rescue 5 (NOW REMOVED - see Session 11):
                   Added Session 10: metastasis + 'beyond regional lymph nodes' -> Correct.
                   REVERSED Session 11: proved factually wrong; metastasis includes N-stage
                   (lymph node) spread; removing the rescue restores correct behaviour.
                3. search_tavily() upgraded: exponential backoff retry on 429 (10s/20s/30s),
                   timeout retry with 5s delay, structured logging per attempt.
                   Fixes citation gap for first ~149 rows caused by rate-limit silently
                   returning [] with no retry.
    Session 11 - Three targeted fixes from 291-row production audit (100% citations):
                Audit findings: 1 false-Incorrect (row 127), 4 false-Corrects (rows 36, 63, 210, 213).
                1. Override E Rescue 5 (bad) REMOVED: the "metastasis beyond regional lymph
                   nodes -> Correct" rescue was factually wrong - metastasis includes regional
                   lymph node spread (N-stage in TNM). Model's original Incorrect verdict was right.
                2. Override A CONTRADICTION_SIGNALS expanded: added "still classified as cancer",
                   "classified as cancer, not", "is still classified", "it is still cancer",
                   "is considered cancerous/cancer". Catches row 213 failure mode where model
                   returns Correct in JSON while reasoning correctly explains Stage 0 IS cancer.
                3. Override E Rescue 5 (new, renumbered): historical lung cancer rarity.
                   Model incorrectly marks row 127 Incorrect by applying modern lung cancer
                   prevalence retroactively. Early-20th-century lung cancer WAS genuinely rare.
                   Fires on: lung cancer + rare + (unlikely to see/encounter | see again |
                   medical student/curiosity | once in a career) -> rescues to Correct.
                Rows 36, 63 (Edwin Smith 3000 BC, Rhazes superlative) were already addressed
                   by Session 10 code (Pattern 8, Override B extension); they appeared here
                   because the chapter was run against an older code version.
    Session 12 - Three targeted fixes from 291-row production audit (100% citations):
                Audit findings: 2 false-Incorrects (rows 176, 268), 1 false-Correct (row 211).
                1. Override D Pattern 9 (NEW): 'metastasis' + 'beyond regional lymph nodes'
                   -> Incorrect. Recurring false-Correct across multiple chapters. Model
                   reasoning is systematically muddled on this pattern. NCI/AJCC definition
                   of metastasis includes N-stage (lymph node) spread, not only M-stage.
                   This assertion incorrectly excludes nodal metastasis from the definition.
                2. Override E Rescue 6 (NEW): Bowen disease / SCC in situ + 'subtype'
                   -> Correct. Model wrongly conflates clinical 'precursor' framing with
                   classificatory 'subtype' designation. WHO Classification of Skin Tumours
                   lists SCC in situ (Bowen disease) as a recognized subtype within SCC.
                3. Override E Rescue 7 (NEW): breast cancer + second + cause of death +
                   United States -> Flagged. Breast IS #2 cause of cancer death in US women
                   (after lung). Assertion is accurate in standard context but imprecise
                   without 'among women'. Model's own reasoning contradicts its Incorrect
                   verdict. Flagged for Stage 2 sex-specific context verification.
    Session 13 - ARCHITECTURAL REDESIGN: root-cause fix for persistent false positives/negatives.
                Root cause analysis across sessions 1-12 identified three structural failures:
                  (1) FALSE CORRECTS: model returns high confidence (0.95-0.99) on wrong
                      claims where it has trained-in biases. Post-hoc Override D/E patching
                      is whack-a-mole - ~2 new errors per 300 assertions, not converging.
                  (2) FALSE INCORRECTS: model over-applies modern/clinical knowledge to
                      historical/classificatory contexts. Override E rescues are reactive.
                  (3) REASONING/JSON MISMATCH: model reasoning contradicts its own JSON
                      verdict (Row 268: reasoning confirms breast IS #2, JSON says Incorrect).
                      Previous Override A only checked reasoning, not analysis field.
                Four architectural changes:
                  A. REDESIGNED STAGE1_PROMPT: Mandatory Flag rules for 7 claim types
                     (A=titles, B=career dates, C=unscoped rankings, D=guideline age thresholds,
                     E=stage survival stats, F=causal attribution %, G=ancient BC dates).
                     Unlike the old "flag if uncertain" suggestion, these are UNCONDITIONAL
                     RULES - model must return Flagged regardless of confidence for these types.
                     This addresses the root cause: known error-category claims can no longer
                     slip through as Correct/Incorrect regardless of model bias.
                  B. CHAIN-OF-THOUGHT JSON OUTPUT: model now returns analysis + concerns +
                     final_verdict + confidence + reasoning. The analysis field is written
                     BEFORE the verdict, reducing reasoning/JSON mismatch. The concerns
                     array surfaces every doubt the model has, even minor ones.
                  C. CONCERNS GATE (NEW): if model's concerns array is non-empty and verdict
                     is Correct or Incorrect, auto-downgrade to Flagged for Review.
                     Principle: any voiced doubt = Stage 2. No doubt = model may clear.
                  D. CONFIDENCE THRESHOLD: raised from 0.95 -> 0.97. Evidence: all
                     false-Correct/Incorrect errors had confidence ≥ 0.95 but were wrong.
                     Tighter threshold further reduces window for confident-wrong model calls.
                  E. OVERRIDE A: now checks BOTH reasoning AND analysis fields for
                     contradiction signals, not just reasoning. Catches the Row-268 pattern
                     where the self-contradiction was in analysis but not in reasoning.
    Session 14 - Five targeted fixes from 284-row production audit (full citation coverage):
                Root cause analysis identified a critical pipeline ordering flaw plus two
                false-Incorrect patterns with no existing rescues:
                  BUG 1 (PIPELINE - CRITICAL): Override D fires only on 'Correct' verdicts.
                    The Session 13 redesigned prompt causes the model to self-flag claims via
                    Mandatory Flag rules (Types A-G), returning 'Flagged for Review' BEFORE
                    Override D runs. Since Override D is guarded by == 'Correct', it is
                    silently skipped for all model-flagged claims. This caused 8 confirmed-wrong
                    claims (Human Factory x2, The Motu Cordis, Virchow 1854, TCGA 2005, CRC
                    stage-I 74%, 90% environmental, USPSTF 50-74) to stay Flagged instead of
                    becoming Incorrect.
                    Fix: change Override D condition from == 'Correct' to
                    in ('Correct', 'Flagged for Review'). Deliberately EXCLUDES 'Incorrect'
                    to prevent P7/P8 (which return Flagged) from downgrading confirmed-wrong
                    Incorrect verdicts.
                  BUG 2 (FALSE INCORRECT - Index 163): assertion optimizer strips phyllodes
                    tumor context. Original: 'Very rarely, the tumor may metastasize' (about
                    phyllodes). Optimized: 'Tumor metastasis is very rare...' (reads as
                    universal claim). No rescue existed. Fix: Override E Rescue 8 - fires on
                    metastas + very rare -> Flagged for Stage 2 context verification.
                  BUG 3 (FALSE INCORRECT - Index 129): temporal projection marked Incorrect.
                    'Expected to increase to 70%' was a valid forward projection when written;
                    now appears wrong because 70% has since been reached. No rescue existed.
                    Fix: Override E Rescue 9 - fires on 'expected to' + survival/rate -> Flagged.
                  BUG 4 (INTERNAL INCONSISTENCY): Override D Pattern 5 reasoning string
                    incorrectly stated 'Virchow became professor at Wurzburg in 1856'.
                    Historical fact: Virchow was already at Wurzburg from 1849; he moved to
                    Berlin in 1856. Fix: corrected reasoning string only, logic unchanged.
                  BUG 5 (PIPELINE SAFETY): confirmed that Override D condition
                    in ('Correct', 'Flagged for Review') - excluding 'Incorrect' - is safe:
                    P7/P8 patterns return Flagged and verified zero match against existing
                    Incorrect rows. No Incorrect verdicts will be downgraded.
    Session 15 - Deep manual audit of 278-row chapter (192 Correct, 11 Incorrect, 75 Flagged).
                All 192 Correct rows audited row-by-row. All 11 Incorrect rows verified.
                Confirmed 2 FALSE CORRECTS and 3 misclassified Incorrects:
                  FALSE CORRECT 1 — "Origin of word cancer credited to Hippocrates"
                    (rows [47], [48] in different chapters). Pattern 11 added to Override D:
                    fires when 'origin ... word ... cancer' is attributed to Hippocrates.
                    FACT: Latin 'cancer' coined by Celsus (~25 BC–50 AD), not Hippocrates.
                    Hippocrates used Greek 'karkinos'. Pattern is narrowly scoped to not
                    fire on correct claims (e.g. Hippocrates described tumors with karkinos).
                  FALSE CORRECT 2 — "NCA 1971 establishing NCI in its current form"
                    (row [107]). Pattern 12 added to Override D: fires when NCA 1971 is
                    paired with 'establishing/established' the NCI.
                    FACT: NCI was established in 1937. NCA 1971 EXPANDED it.
                    Pattern has exception for 'establishing 15 cancer centers' (correct).
                  TCGA 2005 (rows [124], [132]): Changed Pattern 4 from Incorrect → Flagged.
                    NIH's own Dec 14 2005 press release says 'launches'. Full program: 2006.
                    Genuinely ambiguous — Stage 2 should resolve.
                  90% Environmental (row [207]): Changed Pattern 2 from Incorrect → Flagged.
                    Anand et al 2008 (Cancer) peer-reviewed paper defends 90-95% figure under
                    broad 'environmental = non-germline' definition. WHO 30-50% figure refers
                    to preventable cancers (different framing). Stage 2 resolves.
                  Pattern 1 (Stage 0 not cancerous) extended: added 'not considered cancerous'
                    variant to catch row [212]-type assertions.
                  37/37 regression tests passing after all fixes.
    Session 17 - Deep manual audit of 280-row chapter (184 Correct, 12 Incorrect, 84 Flagged).
                All 184 Correct rows verified — 0 false-corrects found.
                All 12 Incorrect rows verified — 1 false-incorrect found.
                  FALSE INCORRECT — Row [40]:
                    Assertion: 'The Edwin Smith Papyrus states there is no treatment
                    for the described disease.'
                    The ESP has 48 cases. Case 45 (bulging masses/breast tumor) explicitly
                    states 'there is no treatment.' The model retrieved the general fact
                    that the ESP offers treatments for some cases, and wrongly treated this
                    as a contradiction. These are about different scopes (one case vs all cases).
                ROOT CAUSE ANALYSIS — Structural recurring bug confirmed:
                  Every false-incorrect across sessions 15-17 shares the same failure mode:
                    SCOPE MISMATCH — the model retrieves a general/related fact and
                    treats it as a contradiction of a specifically-scoped assertion,
                    without verifying that the evidence addresses the SAME scope/metric.
                    [40]:  ESP general treatments vs ESP specific case 'no treatment'
                    [241]: Obesity-type prevalence (40%) vs attributable risk (8%)
                    [207]: WHO consensus (30-50%) vs specific study figure (90%)
                  None of the 7 Mandatory Flag types covered this failure mode.
                  The confidence gate did not catch it because the model found a
                  plausible-looking contradiction and reported high confidence.
                FIX: TRAP 5 — SCOPE MISMATCH added to Stage 1 prompt (Step 4).
                  Instructs model to verify evidence addresses the IDENTICAL scope,
                  definition, population, and metric as the assertion before marking
                  Incorrect. Any possible scope difference -> must use Flagged.
                  Three concrete example patterns provided to guide the model.
                  This is a PROMPT-LEVEL fix targeting the structural root cause,
                  not an Override D/E patch for a specific assertion pattern.
    Session 16 - Deep manual audit of 297-row chapter (199 Correct, 12 Incorrect, 86 Flagged).
                All 199 Correct rows audited individually — 0 false-corrects found.
                All 12 Incorrect rows verified — 1 false-incorrect found.
                  FALSE INCORRECT — Row [241]:
                    Assertion: "Excess body fat is associated with 13 types of cancer
                    and accounts for approximately 8% of cancer diagnoses."
                    The model retrieved the CDC/ACS statistic that obesity-related cancer
                    TYPES account for ~40% of all diagnoses, and incorrectly treated this
                    as a contradiction of the 8% figure. These measure different things:
                      40% = fraction of diagnoses in cancer TYPES associated with obesity
                      8%  = Population Attributable Fraction (PAF): fraction of diagnoses
                            whose occurrence is CAUSALLY ATTRIBUTABLE to excess body weight
                            (IARC Lauby-Secretan et al 2016; ACS Cancer Facts 2021)
                    Fix: Override E Rescue 10 — fires on excess body fat + ~8% + diagnoses
                    → returns Correct with full explanation of the two-statistic distinction.
                    Guard conditions are narrow (all three signals required) to avoid false
                    fires on unrelated 8% cancer statistics.
                  All 11 other Incorrect rows confirmed as true errors ✓.
                  All 199 Correct rows confirmed factually accurate ✓.
    """

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    NUMERIC_VERIFY_PROMPT = """You are a medical fact-checker specializing in numerical accuracy.

ASSERTION: {assertion}

This assertion contains a specific number, percentage, or year.
Using your medical knowledge, is this specific statistic/date accurate?

Answer ONLY with JSON:
{{"accurate": true, "issue": ""}}
or
{{"accurate": false, "issue": "one sentence describing the inaccuracy"}}

Rules:
- If the number is wrong by more than rounding, say false.
- If the date is wrong, say false.
- If you are not certain, say true (give benefit of doubt).
- Do not flag directional claims ("more than", "approximately") as false unless clearly wrong."""

    STAGE1_PROMPT = """You are a medical science fact-checker for a student oncology textbook.

TASK: Reason through this assertion carefully, then classify it.

ASSERTION:
{assertion}

EVIDENCE (from web search - may be empty):
{evidence}

==========================================
STEP 1 - READ THESE MANDATORY RULES FIRST
==========================================

The following claim types MUST receive "Flagged for Review" - NO EXCEPTIONS.
Do not attempt Correct or Incorrect for these, regardless of your confidence.
A human expert will verify all Flagged items in Stage 2.

  MANDATORY FLAG - TYPE A: SPECIFIC PUBLICATION TITLES
    Any assertion citing a specific book or paper title (words in quotation marks).
    Exception: these well-verified titles may be cleared if factually used correctly:
      "De Motu Cordis", "De humani corporis fabrica", "Die krankhaften Geschwülste",
      "Die Entwicklungsgeschichte des Krebses", "Hallmarks of Cancer"
    Any other specific title -> Flag immediately without analysis.

  MANDATORY FLAG - TYPE B: CAREER APPOINTMENT / PROFESSORSHIP DATES
    Any assertion stating the specific year a named person became a professor,
    was appointed to a position, or received a role (e.g. "became professor in 18XX").
    These dates are frequently wrong by 1-7 years in training data.
    Exception: these specific facts are verified and may be cleared:
      Virchow professor at Würzburg 1849; Virchow Berlin 1856
      Vesalius published De humani corporis fabrica 1543
      Harvey published De Motu Cordis 1628

  MANDATORY FLAG - TYPE C: POPULATION-SCOPED RANKINGS WITHOUT EXPLICIT SCOPE
    Any ranking claim (second/third/leading/most common cause/type) about cancer
    where the population denominator is ambiguous (e.g. "in women" vs "overall US").
    Example: "breast cancer is the second leading cause of cancer death in the US"
    is TRUE for women, FALSE for all persons - the scope determines correctness.
    Flag unless the claim explicitly states its population scope.
    Exception: rankings that are unambiguous in any scope (e.g. "tobacco is the
    leading cause of cancer deaths worldwide among men") may be cleared.

  MANDATORY FLAG - TYPE D: CLINICAL GUIDELINE AGE THRESHOLDS
    Any claim stating a specific age threshold for a screening guideline
    (USPSTF, NCCN, WHO, ACS screening recommendations with ages).
    Guidelines update frequently and model training data lags real-world updates.
    Exception: only the following specific verified current guidelines may be cleared:
      USPSTF mammography: age 40+ (2024 guideline - NOT the old 50-74 range)
      USPSTF CRC screening: age 45+ for average-risk adults

  MANDATORY FLAG - TYPE E: EXACT STAGE-SPECIFIC SURVIVAL STATISTICS
    Any claim citing a specific numeric survival rate (%) linked to a cancer stage.
    These figures change with each SEER data update and are frequently outdated.
    Exception: only these verified SEER figures may be cleared:
      CRC stage I ~90% 5-year survival; CRC stage IV ~14%
      Breast cancer stage I ~99% (NOT 100%)

  MANDATORY FLAG - TYPE F: CAUSAL ATTRIBUTION PERCENTAGES
    Any claim stating that X% of cancers are caused by / attributable to a factor.
    These are frequently overstated or based on older literature.
    Exception: only these verified consensus figures may be cleared:
      Tobacco causes ~90% of lung cancer in men, ~80% in women (IARC consensus)
      Environmental/lifestyle factors: ~30-50% of all cancers (NOT 90%+)

  MANDATORY FLAG - TYPE G: ANCIENT BC DATES FOR MEDICAL DESCRIPTIONS
    Any claim citing a specific BC date for a medical text or description
    where the date is not a well-known historical figure's lifespan.
    Exception: these are well-established and may be cleared:
      Hippocrates lived 460-377 BC; Galen lived ~130-200 AD
      Celsus lived ~25 BC-50 AD; Ebers Papyrus ~1500 BC

==========================================
STEP 2 - CLASSIFICATION RULES FOR ALL OTHER CLAIMS
==========================================

"Correct" - Use ONLY when ALL of the following are true:
  • The assertion is not covered by any Mandatory Flag type above
  • It matches established medical/scientific consensus unambiguously
  • Your confidence is ≥ 0.97 based on multiple consistent sources
  • The claim is not a matter of framing, scope, or classification system

"Incorrect" - Use ONLY when ALL of the following are true:
  • The assertion is not covered by any Mandatory Flag type above
  • It directly contradicts established medical facts with certainty
  • The error is unambiguous - not a matter of framing, scope, or classification
  • Your confidence is ≥ 0.97

"Flagged for Review" - Use in ALL other cases, including when:
  • Any Mandatory Flag type (A-G above) applies - NO EXCEPTIONS
  • Your confidence is below 0.97 in either direction
  • The claim involves a "usually/typically/often" qualifier where you might
    mark Incorrect by citing exceptions (the hedge covers exceptions)
  • Historical claims where modern knowledge might be applied retroactively
  • Any claim where your analysis in Step 3 reveals a concern or ambiguity
  • If you find yourself writing a long justification - it should be Flagged

CRITICAL PRINCIPLE - WHEN IN DOUBT, FLAG:
  Stage 1 is a HIGH-SENSITIVITY filter. A correct claim flagged costs nothing
  (Stage 2 will clear it). A wrong claim passed as Correct reaches students unchecked.
  Every sentence of doubt in your analysis is a signal to Flag.

==========================================
STEP 3 - KNOWN FACTS (never contradict these)
==========================================

BRAIN TUMORS:
- Brain metastases ARE the most common brain tumor OVERALL (~10:1 over primaries).

RADIATION:
- UV = NON-IONIZING (WHO/IARC). Ionizing = X-rays, gamma, alpha/beta, neutrons.
- "Non-ionizing includes EMF and UV, the latter causes chromosomal damage" = CORRECT.

CANCER STAGING AND DEFINITION:
- Stage 0 / carcinoma in situ IS cancer (non-invasive). "Not cancerous" = INCORRECT.
- Metastasis = spread of cancer from its original site to another body part.
  This INCLUDES spread to regional lymph nodes (N-stage). NOT only spread beyond nodes.

CANCER EPIDEMIOLOGY:
- Tobacco = leading cause of cancer deaths globally. Causes ~90% lung cancer in men.
- Cancer was 8th leading cause of death in US at start of 20th century. CORRECT.
- Early 20th century: lung cancer WAS genuinely rare - medical students might go
  entire careers without seeing it. Do NOT apply modern prevalence retroactively.

HALLMARKS OF CANCER:
- Hanahan & Weinberg 2000 paper: EXACTLY SIX hallmarks. CORRECT for that paper.
- 2011 update expanded to ten. "Six hallmarks" without year qualifier = CORRECT.

ANATOMY / PHYSIOLOGY:
- Systemic venous blood -> right heart -> LUNGS (first capillary bed). CORRECT.
- Cori cycle: cancer cell lactic acid recycled to glucose in liver. CORRECT.

HISTORY OF MEDICINE:
- Judge historical claims on HISTORICAL accuracy, not modern validity.
  Humoral theory was genuinely believed - do NOT mark it wrong because obsolete.
- Gerard of Cremona translated Canon into Latin in 12th century (died 1187 AD).
  Any "13th century" attribution = INCORRECT.
- Chosroes I sent Perzhoe to India. Jorjani black bile cancer. Remak + Virchow
  cell division. All CORRECT historical facts.

SPECIFIC VERIFIED FACTS:
- TCGA launched 2006, NOT 2005.
- NCI established 1937. 1971 Act EXPANDED it, did not establish it.
- De humani corporis fabrica = correct Vesalius title (not "Human Factory").
- De Motu Cordis = correct Harvey abbreviation (not "The Motu Cordis").

==========================================
STEP 4 - HOW TO HANDLE COMMON TRAPS
==========================================

TRAP 1 - QUALIFIER BLINDNESS:
  "Usually/typically not life-threatening" = CORRECT. Do not mark Incorrect by
  citing edge cases. The qualifier "usually" explicitly covers exceptions.

TRAP 2 - CLASSIFICATORY vs CLINICAL TERMS:
  A term can be correct in one system but different in another.
  "SCC in situ / Bowen disease as a subtype of SCC" = CORRECT per WHO classification,
  even though it is also called a "precursor" clinically. When classificatory and
  clinical terminology diverge, Flag rather than marking Incorrect.

TRAP 3 - MODERN KNOWLEDGE APPLIED RETROACTIVELY:
  A historical claim about what physicians believed or observed in the past
  is NOT made wrong by modern medical advances. Judge it on historical accuracy.

TRAP 4 - REASONING/VERDICT MISMATCH:
  Before writing your final_verdict, re-read your analysis field.
  If your analysis says the claim is true but you're about to write Incorrect,
  or vice versa - reconcile. Your analysis and verdict must agree.

TRAP 5 - SCOPE MISMATCH (most common cause of false Incorrect verdicts):
  Before marking Incorrect, ask: "Is my evidence about the EXACT SAME scope,
  case, population, definition, and metric as the assertion?"
  A general fact about a topic does NOT contradict a specific claim about a
  subset of that topic. If there is ANY possibility your evidence is addressing
  a different scope than the assertion, you MUST use Flagged for Review.

  Common scope mismatch patterns that must trigger Flagged, not Incorrect:
    - Assertion about a SPECIFIC CASE in a text vs evidence about the text in general.
      Example: a text states no treatment for ONE described disease; finding that
      the same text offers treatments for OTHER cases does NOT contradict this.
    - Assertion about ATTRIBUTABLE RISK (% of cancers CAUSED BY a factor) vs
      evidence about PREVALENCE OF TYPE (% of cancers of a type ASSOCIATED WITH factor).
      These are different statistics and do not contradict each other.
    - Assertion about ONE STUDY'S FIGURE vs a DIFFERENT STUDY'S CONSENSUS figure.
      Different studies, populations, and definitions produce different valid numbers.
    - Assertion scoped to a SPECIFIC POPULATION vs evidence about a DIFFERENT POPULATION.

  THE RULE: If your evidence could be about a different scope, definition, or
  metric than the assertion - Flag it. Never mark Incorrect unless your evidence
  directly contradicts the identical scope and metric as the assertion.

==========================================
OUTPUT FORMAT
==========================================

Return JSON only - no markdown, no commentary outside the JSON object:
{{
  "analysis": "Step-by-step reasoning: does this hit a Mandatory Flag type? What does the evidence say? Any concerns?",
  "concerns": ["list every doubt, ambiguity, or qualification - empty array if none"],
  "final_verdict": "Correct | Incorrect | Flagged for Review",
  "confidence": 0.00,
  "reasoning": "one concise sentence explaining the verdict for the output report"
}}

RULES FOR THE OUTPUT:
- analysis: write this BEFORE deciding final_verdict - it is your thinking
- concerns: any item in concerns list is a reason to lean toward Flagged
- confidence: your certainty that final_verdict is correct (0.0-1.0)
- If confidence < 0.97, use "Flagged for Review" - not Correct or Incorrect
- If any Mandatory Flag type A-G applies, final_verdict MUST be "Flagged for Review"
- Your analysis and final_verdict must be logically consistent with each other"""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: Dict[str, Any]):
        load_dotenv()
        self.cfg = config

        gemini_key = os.getenv("GEMINI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        if not gemini_key or not tavily_key:
            raise ValueError(
                "CRITICAL: GEMINI_API_KEY or TAVILY_API_KEY missing in environment."
            )

        self.client = genai.Client(
            api_key=gemini_key,
            http_options={"api_version": "v1beta"},
        )
        self.model = config.get("MODEL_NAME", "gemini-2.0-flash")
        self.tavily_key = tavily_key

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def read_chapter(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_json_loads(self, text: str):
        if not text or not isinstance(text, str):
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {}

    def generate_with_retry(self, prompt_text: str, max_retries: int = 5):
        import concurrent.futures
        timeout_secs = self.cfg.get("GEMINI_TIMEOUT", 45)

        for attempt in range(max_retries):
            def _call():
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt_text,
                    config={"response_mime_type": "application/json"},
                )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                try:
                    return future.result(timeout=timeout_secs)
                except concurrent.futures.TimeoutError:
                    print(
                        f"    [TIMEOUT] attempt {attempt+1}/{max_retries} "
                        f"exceeded {timeout_secs}s — retrying."
                    )
                    future.cancel()
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(5)
                except Exception as e:
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        wait = (attempt + 1) * self.cfg.get("BACKOFF_FACTOR", 20)
                        print(f"    [RATE LIMIT] waiting {wait}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                        continue
                    return None
        return None
    
    def _dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in items or []:
            if not isinstance(x, str):
                continue
            x = x.strip()
            if not x:
                continue
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    # ------------------------------------------------------------------
    # Override A: Contradiction detector
    #
    # Fires when the model's own reasoning contradicts its Correct verdict.
    # Two layers:
    #   1. Substring signals - each validated for zero false positives on production data.
    #      Session 8 changes:
    #        - Replaced bare "was not " with scoped "but it was not" / "was not correct" /
    #          "was not accurate" - the bare form fired on "the term 'cancer' was not used"
    #          (F5), which is supportive rather than contradictory reasoning.
    #        - Added: "two years off", "off by ", "slightly off", "outdated", "superseded",
    #          "has since changed", "is now outdated", "is now age", "was updated" -
    #          catches C68 (Virchow 1854, reasoning said "two years off") and C215
    #          (USPSTF 50-74, reasoning said "now outdated as of 2024").
    #   2. Century mismatch detector - fires only when assertion and reasoning each name
    #      an explicit ordinal century AND they differ.
    # ------------------------------------------------------------------

    def _verdict_contradicts_reasoning(
        self, verdict_str: str, reasoning: str, assertion: str = ""
    ) -> bool:
        if verdict_str != "Correct":
            return False
        r = (reasoning or "").lower()

        CONTRADICTION_SIGNALS = [
            # Core negation signals
            "is largely correct",
            "was not a leading",
            "but it is not a ",
            "but it was not",           # scoped replacement for bare "was not "
            "was not correct",          # explicit correction
            "was not accurate",         # explicit correction
            "is not the ",
            "but the year",
            "but the date",
            "but nci was",
            "but the nci",
            "but was established",
            "not supported by",
            "is incorrect",
            "is wrong",
            "inaccurate",
            "not accurate",
            "but it is not",
            "not entirely correct",
            "but the nci was established",
            "while not correct",        # TIGHTENED: was "while not " (caused false flag on
            "while not accurate",       # "while not as common as" - a qualification, not
            "while not supported",      # contradiction). Now scoped to explicit corrections.
            "though not correct",
            "though not accurate",
            "not 'rare'",
            "not rare",
            # Session 8: year/date admission signals
            "two years off",            # C68: "although it is two years off"
            "off by two",
            "slightly off",
            "off by ",                  # generic "off by N years"
            "not exactly correct",
            "though incorrect",
            "though it is wrong",
            "though it is incorrect",
            # Session 8: stale/outdated guideline signals
            "outdated",                 # C215: "though it is now outdated as of 2024"
            "no longer current",
            "superseded",
            "has since changed",
            "is now outdated",
            "is now age",               # "is now age 40" - guideline changed
            "was updated",
            # Session 9: additional USPSTF/guideline update phrases
            "though the guidelines",    # "though the guidelines have since been updated"
            "have since been updated",  # explicit admission that something changed
            "has since been updated",
            "although the guidelines",  # "although the guidelines were updated in 2024"
            "guidelines were updated",
            "guidelines have been updated",
            "updated in 20",            # catches "updated in 2024", "updated in 2023", etc.
            # Session 11: model admits classification/status contradiction while returning Correct
            "still classified as cancer",       # Row 213: model says "still classified as cancer"
            "classified as cancer, not",        # "classified as cancer, not 'not cancerous'"
            "is still classified",              # generalised form of above
            "it is still cancer",               # direct classification contradiction
            "is considered cancerous",          # "it is considered cancerous" contradicts "not cancerous"
            "is considered cancer",             # same pattern
        ]

        if any(p in r for p in CONTRADICTION_SIGNALS):
            return True

        if assertion and self._detect_century_mismatch(assertion, reasoning):
            return True

        return False

    def _detect_century_mismatch(self, assertion: str, reasoning: str) -> bool:
        """
        Returns True only when assertion and reasoning each contain an explicit
        ordinal century reference AND those centuries differ.
        Requires BOTH texts to have a century AND for them to mismatch.
        """
        CENTURY_RE = re.compile(
            r"\b(1[0-9]|2[0-2])(?:st|nd|rd|th)\s+century\b", re.IGNORECASE
        )

        def extract_century(text: str) -> Optional[int]:
            m = CENTURY_RE.search(text)
            return int(m.group(1)) if m else None

        a_c = extract_century(assertion)
        r_c = extract_century(reasoning)
        return a_c is not None and r_c is not None and a_c != r_c

    # ------------------------------------------------------------------
    # Override B: Absolute-language veto
    #
    # Tightened vs original: removed \\ball\\b, \\bonly\\b, \\bsole\\b that
    # fired on "all known cancers", "second only to", "the only study".
    # "The main cause" intentionally NOT flagged - tobacco as main cause
    # of cancer deaths is factually accurate per WHO.
    # ------------------------------------------------------------------

    def _contains_absolute_language(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        ABSOLUTE_PATTERNS = [
            r"\balways\b",
            r"\bexclusively\b",
            r"\buniversally\b",
        ]
        if any(re.search(p, lowered) for p in ABSOLUTE_PATTERNS):
            return True
        # Scoped 'only': fire on universal causal/treatment claims
        if re.search(r"\bonly (treatment|therapy|cure|way|cause|source)\b", lowered):
            return True
        # Fire if assertion starts with 'only' but NOT a definitional classification
        if re.search(r"^only\b", lowered) and not re.search(
            r"^only (malignant|benign|cancerous|invasive)", lowered
        ):
            return True
        # Scoped superlative: "most important ... in the [Islamic/ancient/world/history/medicine]"
        # Fires on unverifiable historical ranking claims ("Rhazes was the most important
        # specialist in the Islamic world"). Cannot be certified without systematic
        # scholarly review - route to Flagged for Stage 2.
        # Guard: does NOT fire on accepted epidemiology superlatives like "most common cause"
        # - only fires when "in the [world/history/medicine]" follows "most important".
        if re.search(r"\bmost important\b", lowered) and re.search(
            r"\bin the\s+(?:islamic|arab|persian|ancient|medieval|modern)?\s*(?:world|history|medicine|science)\b",
            lowered,
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # Override F: Structural high-risk pattern detector
    #
    # Proactively flags CATEGORIES of claims that have historically produced
    # confident-wrong errors across all production sessions - regardless of
    # model verdict or confidence score.
    #
    # This is the primary architectural defence against whack-a-mole patching.
    # Instead of adding one Override D/E per error, we flag entire RISK CATEGORIES.
    #
    # Error taxonomy from sessions 1-12 (13 total errors analysed):
    #   DEFINITION errors (3x):   ambiguous scope -> flag ranking/definition claims
    #   CLASSIFICATION (2x):      clinical vs classificatory framing -> flag
    #   HISTORICAL (2x):          modern knowledge retroactively applied -> flag
    #   SUPERLATIVE/RANKING (2x): scope-dependent rankings -> flag
    #   GUIDELINE (1x):           stale guidelines -> flag
    #   HEDGE (1x):               model ignores qualifier -> guard in prompt
    #   TITLE (1x):               publication title errors -> flag
    #   ANCIENT_DATE (2x):        contested ancient dates -> flag
    #
    # Returns a string (the flag reason) if the claim should be flagged, else None.
    # ------------------------------------------------------------------

    def _structural_high_risk_check(self, claim: str) -> Optional[str]:
        """
        Returns a flag-reason string if the claim matches a structural high-risk
        pattern. Returns None if the claim does not match any high-risk pattern
        and can proceed to normal override processing.
        """
        c = claim.lower()

        # ── CATEGORY 1: SCOPE-AMBIGUOUS RANKING CLAIMS ──────────────────────
        # Rankings like "most common", "leading cause", "second most" are only
        # valid for a specific denominator (women vs all, primary vs total).
        # When the scope is not explicitly stated, the assertion is ambiguous.
        # Historical production errors: breast cancer #2 US, Rhazes "most important"

        # Epidemiological rankings without explicit population scope
        # Guard: allow clearly-scoped claims ("most common in women", "in the US overall")
        if re.search(r"\b(?:second|third|fourth|2nd|3rd)\s+(?:leading|most common)\s+cause", c):
            # Only flag if population scope is ambiguous (no explicit subgroup mentioned)
            if not re.search(r"\bwomen\b|\bmen\b|\bmale\b|\bfemale\b|\boverall\b", c):
                return (
                    "[Structural flag - ranking scope] Comparative ranking claim "
                    "('second/third leading cause') without explicit population scope (women vs all). "
                    "Scope-dependent rankings are auto-flagged for Stage 2 verification."
                )

        # ── CATEGORY 2: SPECIFIC PUBLICATION TITLES ─────────────────────────
        # Title errors are a recurring source of false-Incorrect verdicts.
        # Examples: "Human Factory", "The Motu Cordis", "Die Entwicklungsgeschichte".
        # Any claim asserting a specific book/paper title should be flagged if
        # the claimed title does not match a known correct title.
        # Guard: known-correct titles (De Motu Cordis, De humani corporis fabrica) pass.
        # We flag UNUSUAL title forms that don't match Latin/standard abbreviations.
        if re.search(r"published|titled|entitled|\"[^\"]{5,}\"", c):
            # Check for known-wrong titles specifically
            if re.search(r"human factory", c):
                return None  # Override D Pattern handles this correctly as Incorrect
            if re.search(r"the motu cordis", c):
                return None  # Override D handles this
            # For any claim with a quoted multi-word title we haven't seen before, flag it
            titles = re.findall(r'"([^"]{5,})"', claim)
            if titles:
                # Known-correct titles that should pass through
                known_correct_titles = [
                    "de motu cordis",
                    "de humani corporis fabrica",
                    "exercitatio anatomica",
                    "zellularpathologie",
                    "die krankhaften",
                    "hallmarks of cancer",
                ]
                for title in titles:
                    if not any(k in title.lower() for k in known_correct_titles):
                        return (
                            f"[Structural flag - publication title] Claim cites a specific "
                            f"publication title (\"{title}\") that requires verification. "
                            f"Title-exactness errors are a recurring source of false verdicts."
                        )

        # ── CATEGORY 3: SPECIFIC HISTORICAL DATES / POSITIONS ───────────────
        # Claims stating the specific year a person became a professor, was appointed,
        # died, or published are only reliable to ~± a few years in model knowledge.
        # When the claim combines a person name + a specific year + a role/event:
        # flag for Stage 2 date verification.
        # Guard: known-verified facts from KNOWN FACTS section pass through.

        KNOWN_VERIFIED_DATES = [
            # (person_pattern, year, context_pattern)
            (r"virchow",    r"1849|1856|1821|1902|1845|1847|1863", None),
            (r"vesalius",   r"1514|1543|1564",                     None),
            (r"harvey",     r"1628",                               r"de motu|motu cordis|circulation"),
            (r"hippocrates",r"460|377|370",                        None),
            (r"galen",      r"130|200",                            None),
            (r"celsus",     r"25|50",                              None),
            (r"cremona",    r"1187",                               None),
            (r"nci",        r"1937",                               None),
            (r"tcga",       r"2006",                               None),
            (r"virchow",    r"1854",                               r"book|publication|handbuch|pathologie"),
        ]

        # Check if claim has a person name + specific year + role/event
        has_person_year = (
            re.search(
                r"\b(?:virchow|vesalius|harvey|hippocrates|galen|celsus|"
                r"paracelsus|avicenna|rhazes|al-razi|jorjani|chosroes|perzhoe|"
                r"cremona|hanahan|weinberg|ramazzini|pott|lister|semmelweis)\b",
                c,
            )
            and re.search(r"\b(?:19|20|18|17|16|15|14|13|12|11|10)\d{2}\b", c)
            and re.search(
                r"\bprofessor\b|\bappointed\b|\bbecame\b|\bfounded\b|\bpublished\b|"
                r"\bdiscovered\b|\bborn\b|\bdied\b|\bestablished\b|\bgraduated\b",
                c,
            )
        )

        if has_person_year:
            # Check if this is a known-verified date pair
            is_verified = False
            for person_pat, year_pat, ctx_pat in KNOWN_VERIFIED_DATES:
                if (
                    re.search(person_pat, c)
                    and re.search(year_pat, c)
                    and (ctx_pat is None or re.search(ctx_pat, c))
                ):
                    is_verified = True
                    break

            if not is_verified:
                return (
                    "[Structural flag - historical date/appointment] Claim specifies a precise "
                    "year for a historical figure's appointment/publication/event that has not "
                    "been pre-verified. Historical date-precision errors are a recurring source "
                    "of false verdicts. Routed to Stage 2 for verification."
                )

        # ── CATEGORY 4: UPDATED CLINICAL GUIDELINES ─────────────────────────
        # Any claim about clinical screening thresholds, age recommendations, or
        # guideline-issuing body recommendations may be outdated.
        # The USPSTF, NCCN, AJCC, WHO update guidelines regularly.
        # Already handled by Override D Pattern 6 for USPSTF mammography.
        # This adds a general net for any guideline claim.
        if re.search(
            r"\b(?:uspstf|nccn|ajcc|who recommends|asco recommends|"
            r"guidelines? (?:recommend|state|suggest)|"
            r"screening (?:guideline|recommendation)|"
            r"(?:recommend|advise)s? (?:screening|mammograph|colonoscop))\b",
            c,
        ) and not re.search(r"uspstf|mammograph", c):
            # The USPSTF mammography case is already handled by Override D Pattern 6.
            # This catches OTHER guideline claims not individually patched.
            return (
                "[Structural flag - clinical guideline] Claim cites a specific clinical "
                "guideline recommendation that may have been updated since model training. "
                "Guideline-currency errors are a recurring source of false Correct verdicts."
            )

        # ── CATEGORY 5: ANCIENT DATE CLAIMS (BC) ────────────────────────────
        # Claims about specific ancient BC dates (e.g. 3000 BC, 1700 BC, 1500 BC)
        # for cancer descriptions or medical history are subject to scholarly debate.
        # Already covered for Edwin Smith 3000 BC (Override D Pattern 8) and 1700 BC
        # (Pattern 7). This adds a general net for any claim with a BC date that is
        # not a well-known historical figure's dates (Hippocrates 460-377 BC etc.)
        # NOTE: Does NOT fire on Hippocrates/Galen/Celsus dates (well-established).
        if re.search(r"\b\d{3,4}\s*bc\b", c, re.IGNORECASE):
            # Allow known-established BC dates for historical figures
            known_bc_persons = [
                r"hippocrates",
                r"galen",      # born 129-130 AD but sometimes expressed as BC context
                r"celsus",
                r"dioscorides",
            ]
            # Allow known-well-established date ranges
            known_bc_ranges = [
                r"460.{0,5}370|460.{0,5}377",   # Hippocrates lifespan
                r"25\s*bc.{0,10}50\s*ad",        # Celsus
            ]
            is_known_bc = (
                any(re.search(p, c) for p in known_bc_persons)
                or any(re.search(p, c) for p in known_bc_ranges)
                or re.search(r"ebers papyrus", c)   # 1500 BC is well-accepted
            )
            if not is_known_bc:
                return (
                    "[Structural flag - ancient BC date] Claim cites a specific ancient BC "
                    "date for a cancer/medical description. Ancient date claims are contested "
                    "and require expert verification. Routed to Stage 2."
                )

        return None

    # ------------------------------------------------------------------
    # Override C: Numeric self-check
    #
    # Triggers only on:
    #   - Exact single percentages (e.g. 48%) - NOT ranges (10-26%)
    #   - Post-1930 specific years NOT inside a range and NOT approximate
    # Does NOT trigger on: ranges, multipliers, approximate counts, ancient dates.
    # ------------------------------------------------------------------

    def _has_specific_numbers(self, text: str) -> bool:
        # Strip percentage ranges so exact % check doesn't fire on them
        cleaned = re.sub(r"\d+\s*(?:[-\u2013]|to)\s*\d+\s*%", "", text)
        cleaned = re.sub(
            r"\d+\s*(?:[-\u2013]|to)\s*\d+\s*(?:times|fold|x)\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        has_exact_pct = bool(re.search(r"\b\d{1,3}\s*%", cleaned))

        has_year_range = bool(
            re.search(r"\b(?:19|20)\d{2}\s*[-\u2013]\s*(?:19|20)\d{2}\b", text)
        )
        has_approx = bool(
            re.search(
                r"(?:over|more than|approximately|about|up to|at least|almost)\s+\d",
                text,
                re.IGNORECASE,
            )
        )
        has_modern_year = bool(re.search(r"\b(?:19[3-9]\d|20[0-2]\d)\b", text))
        specific_modern_year = has_modern_year and not has_year_range and not has_approx

        return has_exact_pct or specific_modern_year

    def _numeric_self_check(self, claim: str) -> dict:
        """Secondary model call for exact-number assertions. Fails open (benefit of doubt)."""
        p = self.NUMERIC_VERIFY_PROMPT.format(assertion=claim)
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=p,
                config={"response_mime_type": "application/json"},
            )
            result = self._safe_json_loads(resp.text if resp else "")
            if isinstance(result, dict) and "accurate" in result:
                return result
        except Exception:
            pass
        return {"accurate": True, "issue": ""}

    # ------------------------------------------------------------------
    # Override D: Hard-coded false-Correct catches
    #
    # Deterministic regex patterns for confirmed recurring false-Correct errors.
    # Applied when verdict is 'Correct' OR 'Flagged for Review'.
    # Converts to Incorrect (or Flagged for Review where appropriate).
    # Each pattern validated: zero false positives on production Correct rows.
    #
    # SESSION 14 - BUG 1 FIX: Override D now also runs on 'Flagged for Review'
    # verdicts (in addition to 'Correct'). Root cause: the Session 13 redesigned
    # prompt causes the model to self-flag claims via Mandatory Flag rules (Types
    # A-G), returning 'Flagged for Review' BEFORE Override D can run. Since
    # Override D was previously guarded by == 'Correct', it was silently skipped
    # for all model-flagged claims. This caused 8 confirmed-wrong claims to stay
    # Flagged instead of becoming Incorrect:
    #   - Human Factory title (indices 83, 109)
    #   - The Motu Cordis title (index 70)
    #   - Virchow 1854 professor (index 95)
    #   - TCGA launched 2005 (index 130)
    #   - CRC stage I survival 74% (index 208)
    #   - 90% environmental cancer attribution (index 216)
    #   - USPSTF mammography ages 50-74 (index 268)
    # The condition is intentionally in ('Correct', 'Flagged for Review') and
    # deliberately EXCLUDES 'Incorrect' to prevent P7/P8 (which return Flagged)
    # from downgrading confirmed-wrong Incorrect verdicts. Verified zero matches
    # against existing Incorrect rows for P7 and P8 patterns.
    #
    # Patterns (sessions 6, 8, 9):
    #   1. Stage 0 "not cancerous" / "not yet cancerous"
    #   2. 90%+ environmental/lifestyle cancer causation
    #   3. Colorectal stage I survival ~74%
    #   4. TCGA launched 2005 (actual: 2006)          [Session 8]
    #   5. Virchow professor in 1854 (actual: 1849 Wurzburg / 1856 Berlin)
    #      [Session 8, tightened Session 9, reasoning corrected Session 14 - Bug 4]
    #   6. USPSTF mammography 50-74 (superseded 2024) [Session 9]
    #   7. 1700 BC cancer reference (no source exists) [Session 9]
    #   8. Edwin Smith Papyrus "dated 3000 BC"         [Session 10]
    #   9. Metastasis "beyond regional lymph nodes"    [Session 12]
    #  10. Known-wrong publication titles (Human Factory, The Motu Cordis)
    # ------------------------------------------------------------------

    def _hard_coded_fact_check(self, claim: str) -> Optional[dict]:
        """
        Returns {"verdict": "Incorrect", "reasoning": ...} on match, else None.
        """
        c = claim.lower()

        # Pattern 1: Stage 0 / CIS described as "not cancerous" or "not yet cancerous"
        # Extends to: "not considered cancerous" (variant seen in multiple chapters)
        # FACT: Stage 0 / carcinoma in situ IS classified as cancer by NCI/AJCC —
        # it is a non-invasive cancer. "Not cancerous" and "not considered cancerous"
        # are both wrong phrasings for this concept.
        if (
            ("stage 0" in c or "carcinoma in situ" in c or "in situ" in c)
            and re.search(
                r"\bnot\s+(?:yet\s+)?(?:cancerous|cancer)\b"
                r"|\bnot\s+considered\s+(?:cancerous|cancer|malignant)\b",
                c,
            )
        ):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] Stage 0/carcinoma in situ IS cancer per NCI/AJCC "
                    "(non-invasive form of cancer). 'Not cancerous' / 'not considered "
                    "cancerous' are incorrect — CIS is classified as cancer, not a "
                    "pre-cancerous condition."
                ),
            }

        # Pattern 2: 90%+ environmental/lifestyle cancer causation
        # REVISED Session 14+: Changed from Incorrect → Flagged for Review.
        # RATIONALE: This is a genuine scientific controversy, not a simple error.
        # - WHO/IARC/CDC: ~30-50% of cancers are preventable (modifiable risk factors)
        # - Anand et al 2008 (Cancer, peer-reviewed): "90-95% of cancer cases due to
        #   environmental factors" — here 'environmental' = everything non-germline
        #   (lifestyle, infections, diet, carcinogens, plus non-inherited mutations)
        # - Wu et al 2016 (Nature): ~70-90% attributable to environment broadly defined
        # The two figures refer to different framings: the WHO's PREVENTABLE framing
        # vs the genetics framing (non-inherited = 90%+). Both are published science.
        # Since this cannot be deterministically resolved without context, route to
        # Stage 2 for expert judgment on which definition the textbook intends.
        # EXCEPTION: If "preventable" is the explicit framing, model will usually
        # catch this with Override A (contradiction signals). Pattern 2 now only
        # routes to Flagged, not hard-Incorrect.
        if re.search(r"\b9[05]\s*%|\bover\s+90\s*%", c) and re.search(
            r"environmental|lifestyle|modifiable|preventable", c
        ):
            return {
                "verdict": "Flagged for Review",
                "reasoning": (
                    "[Hard-coded] 90%+ environmental/lifestyle attribution is scientifically "
                    "contested. WHO/IARC/CDC cite ~30-50% preventable cancers; Anand et al "
                    "2008 (Cancer) cites 90-95% 'environmental' (non-germline) — different "
                    "definitional frameworks. Route to Stage 2 for expert clarification of "
                    "which definition the source intends."
                ),
            }

        # Pattern 3: Colorectal stage I survival ~74%
        if (
            re.search(r"colorectal|colon.rectal|\bcrc\b", c)
            and re.search(r"stage\s+i\b", c)
            and re.search(r"7[0-9]\s*%", c)
            and "survival" in c
        ):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] SEER: colorectal stage I 5-year survival ~90%, "
                    "not ~74%. Stated figure significantly understates survival."
                ),
            }

        # Pattern 10: Known-wrong publication titles — Vesalius and Harvey
        # "Human Factory" = wrong translation of De humani corporis fabrica.
        # "The Motu Cordis" = wrong; correct is "De Motu Cordis".
        # These are caught by Override F Category 2 which returns None (bypasses F),
        # trusting Override D to mark them Incorrect. This pattern fulfils that contract.
        if re.search(r"human factory", c) and re.search(r"vesalius|1543", c):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] Vesalius's book is titled 'De humani corporis fabrica' "
                    "(On the Fabric of the Human Body). 'Human Factory' is an incorrect "
                    "colloquial mistranslation of the Latin title."
                ),
            }
        if re.search(r"the motu cordis", c) and re.search(r"harvey|1628", c):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] Harvey's work is abbreviated 'De Motu Cordis', "
                    "not 'The Motu Cordis'. 'The' is incorrect — 'De' is the correct "
                    "Latin preposition meaning 'on/concerning'."
                ),
            }

        # Pattern 4: TCGA launched in 2005 — REVISED: Flagged (not Incorrect)
        # The NIH issued a formal press release on Dec 14, 2005:
        # "NIH Launches Comprehensive Effort to Explore Cancer Genomics" (genome.gov)
        # The full program was funded/operational in 2006, but many published sources
        # (including NIH's own page) use 2005 as the launch date for the pilot.
        # Multiple peer-reviewed papers also cite 2005 as the start of TCGA.
        # This is a genuine date ambiguity (pilot announcement vs full program), not
        # a clear factual error. Route to Stage 2 for expert clarification.
        if re.search(r"cancer genome atlas|tcga", c) and re.search(r"\b2005\b", c):
            return {
                "verdict": "Flagged for Review",
                "reasoning": (
                    "[Hard-coded] TCGA date is ambiguous: NIH's Dec 14, 2005 press release "
                    "used the word 'launches' for the TCGA pilot announcement, while the "
                    "full program was funded/operational in 2006. Multiple published sources "
                    "cite both 2005 and 2006 as the start date. Route to Stage 2 to verify "
                    "which year the textbook's primary source specifically cites."
                ),
            }

        # Pattern 5: Virchow became professor in 1854
        # TIGHTENED: require 'professor', 'became', or 'appointed' in assertion to avoid
        # firing on book titles that contain 'Pathologie' (e.g. 'Handbuch der speciellen
        # Pathologie und Therapie' published 1854 - correct date, different context).
        # SESSION 14 BUG 4 FIX: corrected reasoning string. Previous version incorrectly
        # stated 'Virchow became professor at Wurzburg in 1856'. Historical fact: Virchow
        # was already at Wurzburg from 1849; he moved to Berlin in 1856. Neither
        # appointment was in 1854. Reasoning text corrected; logic unchanged.
        if (
            re.search(r"virchow", c)
            and re.search(r"\b1854\b", c)
            and re.search(r"\bprofessor\b|\bbecame\b|\bappointed\b|\bchair\b", c)
        ):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] Virchow was already professor at Würzburg from 1849. "
                    "He was appointed to Berlin in 1856. Neither appointment was in 1854."
                ),
            }

        # Pattern 6: USPSTF mammography 50-74 (superseded 2024 guideline)
        # USPSTF 2024 updated: biennial mammography from age 40 for all women.
        # The 50-74 range is the old 2016 guideline. Model inconsistently flags this.
        # Hard-code to deterministically catch it regardless of model reasoning.
        if (
            re.search(r"uspstf|u\.s\.\s*preventive\s*services\s*task\s*force", c)
            and re.search(r"mammograph", c)
            and re.search(r"\b50\b|\baged\s+50\b", c)
            and re.search(r"\b74\b|\bto\s+74\b", c)
        ):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] USPSTF 2024 guideline recommends biennial mammography "
                    "starting at age 40, not 50. The 50-74 range reflects the superseded "
                    "2016 recommendation. Current guideline includes all women aged 40+."
                ),
            }

        # Pattern 7: Cancer references dated to 1700 BC
        # No confirmed cancer source from 1700 BC exists. Edwin Smith Papyrus = ~1600 BC.
        # The model incorrectly reasons that '1700 BC ~= 1600 BC' (logically backwards -
        # 1700 BC is 100 years OLDER, not a rounding of 1600 BC).
        # Route to Flagged for Stage 2 expert verification.
        if re.search(r"1700\s*bc", c) and re.search(r"cancer|tumor|tumour|carcinoma", c):
            return {
                "verdict": "Flagged for Review",
                "reasoning": (
                    "[Hard-coded] No confirmed cancer reference from 1700 BC exists. "
                    "Earliest known is Edwin Smith Papyrus (~1600 BC). "
                    "Route to Stage 2 for expert verification."
                ),
            }

        # Pattern 8: Edwin Smith Papyrus "dated 3000 BC"
        # The physical Edwin Smith Papyrus dates to ~1550-1600 BC. The "3000 BC" figure
        # refers to possible authorship of OLDER source texts - not the papyrus date itself.
        # Model reasoning sometimes agrees with 3000 BC (no self-contradiction to catch),
        # so Override A doesn't fire. Override D intercepts it here.
        # Route to Flagged: the claim is historically contested and requires expert review.
        if (
            re.search(r"edwin smith papyrus", c)
            and re.search(r"3000\s*bc", c)
        ):
            return {
                "verdict": "Flagged for Review",
                "reasoning": (
                    "[Hard-coded] The Edwin Smith Papyrus physical manuscript dates to "
                    "~1550-1600 BC, not 3000 BC. The '3000 BC' figure refers to possible "
                    "original authorship of source texts - disputed and not the papyrus date. "
                    "Route to Stage 2 for expert verification."
                ),
            }

        # Pattern 9: Metastasis defined as spread "beyond regional lymph nodes"
        # This is a recurring false-Correct pattern across multiple chapters.
        # FACT: Metastasis is the spread of cancer FROM its primary site TO another body
        # part. Per NCI/AJCC, this includes spread TO regional lymph nodes (N-stage) AND
        # spread to distant organs (M-stage). An assertion defining metastasis as spread
        # 'beyond regional lymph nodes' EXCLUDES N-stage (nodal metastasis), which is wrong.
        # Model reasoning is systematically muddled on this assertion - validating both
        # the wrong definition and the correct counter. Override D enforces Incorrect.
        # Note: This is DIFFERENT from Override E Rescue 5 (removed Session 11) which
        # incorrectly RESCUED this pattern. This Override D actively marks it Incorrect.
        if (
            re.search(r"\bmetastas(?:is|es|ize)\b", c)
            and re.search(r"beyond\s+(?:the\s+)?regional\s+lymph\s+node", c)
        ):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] Defining metastasis as spread 'beyond regional lymph nodes' "
                    "is incorrect. Per NCI/AJCC, metastasis includes spread TO regional lymph "
                    "nodes (N-stage in TNM) as well as distant organs (M-stage). The assertion "
                    "incorrectly excludes nodal metastasis from the definition."
                ),
            }

        # Pattern 11: "Origin of the word cancer" credited to Hippocrates
        # Confirmed FALSE CORRECT across multiple chapters (rows [48], [47] etc).
        # FACT: Hippocrates (Greek physician) used 'karkinos' and 'carcinoma' — Greek terms.
        # The LATIN word 'cancer' was coined by Celsus (~25 BC–50 AD) as a direct translation
        # of the Greek 'karkinos' (meaning crab). Hippocrates originated the concept and
        # the Greek terminology, but NOT the Latin word 'cancer' itself.
        # The same dataset typically includes a correct row (e.g. "Celsus applied the Latin
        # word cancer...") that directly contradicts the Hippocrates attribution.
        # The model is systematically misled by sources (incl. SEER training pages and
        # social-media sites) that loosely credit Hippocrates because his Greek concept
        # was the etymological origin — but the Latin WORD is Celsus's coinage.
        # NARROW MATCH: fires only when 'origin of the word cancer' (or equivalent) is
        # explicitly credited to Hippocrates. Does NOT fire on general claims that
        # Hippocrates described cancer or used karkinos/carcinoma (those are correct).
        if (
            re.search(r"origin.{0,40}word.{0,20}cancer", c)
            and re.search(r"\bhippocrates\b", c)
        ):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] The Latin word 'cancer' was coined by the Roman physician "
                    "Celsus (~25 BC–50 AD) as a translation of the Greek 'karkinos'. "
                    "Hippocrates used 'karkinos'/'carcinoma' (Greek), not the Latin 'cancer'. "
                    "Crediting the origin of the word 'cancer' to Hippocrates conflates the "
                    "Greek concept with the Latin term — Celsus coined the Latin word."
                ),
            }

        # Pattern 12: NCA 1971 "established" the NCI (vs correctly: EXPANDED it)
        # Confirmed FALSE CORRECT across multiple chapters.
        # FACT: The National Cancer Institute was established in 1937 by the National Cancer
        # Institute Act. The National Cancer Act of 1971 dramatically EXPANDED the NCI's
        # authority, budget, and mandate — it did NOT establish the NCI.
        # Assertions claiming NCA 1971 "established the NCI" or "established the NCI in
        # its current form" are factually wrong. The model rationalises this by treating
        # "established in its current form" as equivalent to "significantly expanded" —
        # but the assertion is contradicted by rows in the same datasets that correctly
        # state the NCI was established in 1937.
        # NARROW: fires only when NCA/NCA 1971 is explicitly paired with "establishing"
        # or "established" the NCI. Does NOT fire on "NCA expanded the NCI" (correct).
        if (
            re.search(r"national cancer act|nca\b", c)
            and re.search(r"1971", c)
            and re.search(r"\bestab(?:lished|lishing)\b", c)
            and re.search(r"national cancer institute|nci\b", c)
            and not re.search(
                r"estab(?:lished|lishing).{0,60}(?:15|cancer center|seer|program)",
                c,
            )
        ):
            return {
                "verdict": "Incorrect",
                "reasoning": (
                    "[Hard-coded] The National Cancer Institute was established in 1937 by "
                    "the National Cancer Institute Act, NOT by the National Cancer Act of 1971. "
                    "The 1971 NCA dramatically expanded the NCI's authority, budget, and "
                    "mandate, but the NCI itself had existed since 1937. Assertions stating "
                    "the NCA 1971 'established the NCI' are factually incorrect."
                ),
            }

        return None

    # ------------------------------------------------------------------
    # Override E: Hard-coded false-Incorrect rescue
    #
    # Deterministic regex for confirmed recurring false-Incorrect verdicts -
    # cases where the model marks correct facts as wrong.
    # Applied when verdict == Incorrect. Converts to Correct (or Flagged).
    # Runs LAST so it can also rescue mis-fired overrides A/B/C/D.
    #
    # Rescues validated on production data:
    #   1. Brain mets = most common brain tumor (persistent 3-run false-Incorrect)
    #   2. Six hallmarks of cancer for H&W 2000 (model penalises for 2011 update)
    #   3. Historical black bile / Jorjani description (modern science applied to history)
    #   4. Benign tumors 'usually not life-threatening' (model ignores 'usually' qualifier)
    #      [Session 10]
    #   5. Metastasis "beyond regional lymph nodes" rescue REMOVED Session 11
    #      (was factually wrong - metastasis includes lymph node spread; see below)
    #   6. Historical lung cancer rarity - early 20th century (Session 11):
    #      model applies modern prevalence knowledge to historical scarcity claim.
    #   7. Breast cancer second cause of death US -> Flagged (Session 12):
    #      assertion accurate for women but imprecise without 'among women' scope.
    #   8. Context-stripped "very rare metastasis" claim -> Flagged (Session 14 - Bug 2):
    #      assertion optimizer strips tumor-type context (e.g. phyllodes), causing model
    #      to evaluate a specific tumor claim as a universal claim and mark Incorrect.
    #   9. Temporal survival rate projection -> Flagged (Session 14 - Bug 3):
    #      'expected to increase to X%' projections valid at time of writing are marked
    #      Incorrect once the milestone has been reached. Temporal artifacts require
    #      date-context verification, not a flat Incorrect verdict.
    # ------------------------------------------------------------------

    def _hard_coded_rescue(self, claim: str) -> Optional[dict]:
        """
        Returns {"verdict": "Correct", "reasoning": ...} on match, else None.
        """
        c = claim.lower()

        # Rescue 1: Brain metastases = most common brain tumor overall
        # Model confuses "most common PRIMARY brain tumor" with "most common overall".
        # FACT: Brain mets outnumber primary tumors ~10:1. Persistent false-Incorrect.
        if (
            re.search(r"brain met(?:astases|astasis|s)\b", c)
            and re.search(r"most common (?:type of )?brain tumor", c)
        ):
            return {
                "verdict": "Correct",
                "reasoning": (
                    "[Hard-coded rescue] Brain metastases ARE the most common brain "
                    "tumor overall (~10:1 over primary tumors per NCI/AANS)."
                ),
            }

        # Rescue 2: Six hallmarks of cancer (H&W 2000 paper)
        # Model marks Incorrect because 2011 update expanded to ten.
        # FACT: The 2000 paper described exactly six - assertion is valid.
        # Guard: does NOT fire if assertion says "current"/"updated"/"2011"/"ten".
        if (
            re.search(r"six\s+(?:biological\s+)?(?:capabilities|hallmarks)", c)
            and re.search(r"hallmark", c)
            and not re.search(r"current|updated|today|now|2011|ten\b|10\b", c)
        ):
            return {
                "verdict": "Correct",
                "reasoning": (
                    "[Hard-coded rescue] H&W 2000 described exactly six hallmarks. "
                    "Assertion is correct for the landmark 2000 paper."
                ),
            }

        # Rescue 3: Historical humoral/medieval cancer descriptions (Jorjani pattern)
        # Model marks Incorrect for using "outdated" humoral theory language.
        # FACT: Historical medical claims describe what physicians believed - judge
        # on historical accuracy, not modern science validity.
        if (
            re.search(r"black bile", c)
            and re.search(r"cancer|carcinoma|swelling", c)
            and re.search(r"scirrhus|pulsation|inflammation|unlike", c)
        ):
            return {
                "verdict": "Correct",
                "reasoning": (
                    "[Hard-coded rescue] Historical description per Islamic/medieval "
                    "medicine. Historical claims judged on historical accuracy, "
                    "not modern science validity."
                ),
            }

        # Rescue 4: Benign tumors "usually not damaging or life-threatening"
        # Model over-corrects by citing exceptions (pituitary adenoma, brain meningioma),
        # ignoring the "usually" qualifier that explicitly accommodates those exceptions.
        # FACT: NCI, ACS, and all major oncology textbooks describe benign tumors as
        # "typically" or "usually" not life-threatening. This IS standard textbook phrasing.
        # The "usually" hedge is doing critical work - don't fire when it's present.
        if (
            re.search(r"\bbenign\b", c)
            and re.search(r"\busually\b|\btypically\b|\bgenerally\b", c)
            and re.search(r"not.{0,20}(?:damaging|life.threatening|harmful)", c)
        ):
            return {
                "verdict": "Correct",
                "reasoning": (
                    "[Hard-coded rescue] 'Usually/typically not life-threatening' is "
                    "standard NCI/ACS/textbook language for benign tumors. The qualifier "
                    "'usually' accommodates exceptions. Marking Incorrect is over-correction."
                ),
            }

        # REMOVED - Rescue 5 (metastasis "beyond regional lymph nodes" -> Correct):
        # Session 11 audit proved this rescue is FACTUALLY WRONG.
        # Standard medical definition of metastasis includes regional lymph node spread
        # (N-stage in TNM). An assertion saying metastasis = "beyond regional lymph nodes"
        # EXCLUDES nodal metastasis from the definition - that is incorrect.
        # The model's original Incorrect verdict for such assertions is right.
        # Rescue removed to restore correct behaviour.

        # Rescue 5 (renumbered): Historical lung cancer rarity - early 20th century
        # Model applies modern awareness of lung cancer prevalence retrospectively and
        # incorrectly marks as Incorrect claims that lung cancer was rare enough to be
        # a medical curiosity. FACT: At the beginning of the 20th century lung cancer
        # was genuinely rare - physicians documented it as a condition medical students
        # might see only once in a career. Multiple historical oncology texts confirm this.
        # Guard: fires only when "rare" + "lung cancer" + historical scarcity framing
        # ("unlikely to see", "medical curiosity", "unlikely to encounter", "see again")
        # is present, ensuring it doesn't fire on modern lung cancer claims.
        if (
            re.search(r"lung cancer", c)
            and re.search(r"\brare\b", c)
            and re.search(
                r"unlikely\s+to\s+(?:see|encounter)|see\s+(?:it\s+)?again|medical\s+(?:student|curiosity)|once\s+in\s+a\s+career",
                c,
            )
        ):
            return {
                "verdict": "Correct",
                "reasoning": (
                    "[Hard-coded rescue] Historical lung cancer rarity claim is accurate. "
                    "In the early 20th century lung cancer was genuinely rare - physicians "
                    "documented it as a condition students might see only once in a career. "
                    "Model incorrectly applies modern prevalence knowledge to historical context."
                ),
            }

        # Rescue 6: Bowen disease / SCC in situ as a histological subtype of SCC
        # Model incorrectly classifies Bowen disease as ONLY a 'precursor' and marks
        # assertions calling it a 'histological subtype' as Incorrect.
        # FACT: The WHO Classification of Skin Tumours lists 'squamous cell carcinoma in situ
        # (Bowen disease)' as a recognized entity within the SCC spectrum - a subtype/variant.
        # While it is also a precursor lesion clinically, the CLASSIFICATORY term 'subtype'
        # is accurate and standard in pathology. The model conflates clinical staging
        # ('precursor' = pre-invasive stage) with pathological classification ('subtype' =
        # variant within a tumor category). Both can be true simultaneously.
        if (
            re.search(r"\b(?:bowen|squamous cell carcinoma in situ)\b", c)
            and re.search(r"\bsubtype\b", c)
        ):
            return {
                "verdict": "Correct",
                "reasoning": (
                    "[Hard-coded rescue] WHO Classification of Skin Tumours lists SCC in situ "
                    "(Bowen disease) as a recognized subtype/variant within the SCC spectrum. "
                    "While clinically a precursor lesion, it is correctly classified as a "
                    "histological subtype of SCC in standard pathology systems."
                ),
            }

        # Rescue 7: Breast cancer as second leading cause of cancer death in the United States
        # Model marks Incorrect with confused reasoning: confirms breast IS #2 behind lung
        # cancer, then adds 'not overall' to invalidate its own logic.
        # FACT: In women - the universal context for this claim in oncology textbooks -
        # breast cancer IS the second leading cause of cancer death after lung cancer.
        # The assertion is standard language from ACS, NCI, Komen, BCRF.
        # Resolution: Convert to Flagged - flagging for 'in women' precision.
        if (
            re.search(r"\bbreast cancer\b", c)
            and re.search(r"\bsecond\b", c)
            and re.search(r"cause of (?:cancer\s+)?death", c)
            and re.search(r"united states", c)
        ):
            return {
                "verdict": "Flagged for Review",
                "reasoning": (
                    "[Hard-coded rescue] Breast cancer IS the second leading cause of cancer "
                    "death among women in the US (after lung cancer) - the assertion is correct "
                    "in the standard clinical context for women. However, without specifying "
                    "'among women', the claim is ambiguous (breast cancer is not #2 in the "
                    "overall US population including men). Flagged for Stage 2 verification."
                ),
            }

        # Rescue 8: Context-stripped "very rare metastasis" claim
        # SESSION 14 BUG 2 FIX: The assertion optimizer can strip tumor-type context from
        # claims about specific tumor types where rare metastasis is medically accurate
        # (e.g. phyllodes tumors, low-grade sarcomas). When context is lost, the claim
        # reads as a universal statement ('Tumor metastasis is very rare') which the model
        # correctly marks Incorrect as a general claim. However, this produces a false
        # Incorrect when the original source was about a specific rare-metastasis tumor type.
        # Resolution: Flagged for Stage 2 context verification rather than flat Incorrect.
        # Guard: fires only on metastas + very rare combination; does not fire on claims
        # with specific organ/tumor-type context that the model can evaluate correctly.
        if (
            re.search(r"\bmetastas", c)
            and re.search(r"\bvery\s+rare\b", c)
        ):
            return {
                "verdict": "Flagged for Review",
                "reasoning": (
                    "[Hard-coded rescue] 'Very rare metastasis' claim may be about a specific "
                    "tumor type (e.g. phyllodes, low-grade sarcoma) whose context was lost "
                    "during assertion optimization. For certain tumor types rare metastasis "
                    "even in malignant grades is medically accurate. "
                    "Flagged for Stage 2 context verification."
                ),
            }

        # Rescue 9: Temporal survival rate projection
        # SESSION 14 BUG 3 FIX: Claims framed as forward projections ('expected to increase
        # to X%', 'expected to reach X%') for survival rates may have been accurate when the
        # textbook was written, even if the projected milestone has since been achieved.
        # The model marks these Incorrect with 'already reached' reasoning, which is
        # technically correct in real-time but inappropriate for evaluating textbook text
        # that was valid at time of writing. These require date-context verification.
        # Resolution: Flagged for Stage 2 temporal context review.
        # Guard: fires only on 'expected to' + increase/reach/improve + survival/rate;
        # does not fire on simple current-state survival statistics.
        if (
            re.search(r"\bexpected\s+to\s+(?:increase|reach|improve|rise|climb)\b", c)
            and re.search(r"(?:survival|rate)\b", c)
        ):
            return {
                "verdict": "Flagged for Review",
                "reasoning": (
                    "[Hard-coded rescue] Forward survival-rate projection ('expected to "
                    "reach/increase to X%') may have been accurate at time of writing even "
                    "if the target has since been met. Temporal projections in textbooks "
                    "require date-context verification. Flagged for Stage 2."
                ),
            }

        # Rescue 10: Excess body fat / obesity → approximately 8% of cancer diagnoses
        # CONFIRMED FALSE INCORRECT across Sessions 15 and 16 (rows [241] in multiple chapters).
        # ROOT CAUSE: The model retrieves the CDC/ACS figure that "obesity-related cancer
        # types account for ~40% of all cancer diagnoses" and interprets it as contradicting
        # the "8%" figure. These are fundamentally DIFFERENT statistics:
        #
        #   ┌─────────────────────────────────────────────────────────────────────┐
        #   │ 40% = fraction of ALL diagnoses that fall into cancer TYPES which    │
        #   │       are KNOWN to be ASSOCIATED with excess body weight.            │
        #   │       (Source: CDC MMWR 2016; Islami et al 2018)                    │
        #   │       i.e. "40% of diagnoses are of a cancer type that obesity       │
        #   │            can contribute to" — NOT that obesity CAUSED 40%          │
        #   │                                                                      │
        #   │ 8%  = fraction of ALL cancer diagnoses whose occurrence is           │
        #   │       CAUSALLY ATTRIBUTABLE to excess body weight.                   │
        #   │       (Source: IARC Working Group / Lauby-Secretan et al 2016;       │
        #   │                ACS Cancer Facts & Figures 2021)                      │
        #   │       i.e. "excess weight is the actual cause for 8% of diagnoses"   │
        #   └─────────────────────────────────────────────────────────────────────┘
        #
        # Both statistics are valid and published. They are NOT contradictory.
        # The 8% is correctly described as "accounts for approximately 8% of cancer
        # diagnoses" — this is the PAF (Population Attributable Fraction) for excess
        # body weight across all cancers.
        #
        # GUARD conditions (must ALL be true to fire):
        #   1. "8 percent" OR "8%" appears in the assertion
        #   2. "cancer" AND "diagnos" (diagnosis/diagnoses) appear
        #   3. Excess body fat / obesity / overweight mentioned
        # This is narrow enough to avoid false fires on unrelated 8% statistics.
        if (
            re.search(r"\b8\s*%|\b8\s+percent\b", c)
            and re.search(r"\bcancer\b", c)
            and re.search(r"\bdiagnos", c)
            and re.search(
                r"\bexcess\s+body\s+(?:fat|weight)\b"
                r"|\bexcess\s+fat\b"
                r"|\bobes(?:e|ity)\b"
                r"|\boverweight\b",
                c,
            )
        ):
            return {
                "verdict": "Correct",
                "reasoning": (
                    "[Hard-coded rescue] The ~8% figure represents the Population Attributable "
                    "Fraction (PAF) — the fraction of ALL cancer diagnoses causally attributable "
                    "to excess body weight (IARC Working Group, Lauby-Secretan et al 2016; ACS "
                    "Cancer Facts 2021). This is distinct from the oft-cited '40% of diagnoses "
                    "belong to obesity-associated cancer TYPES' (CDC 2016). Both statistics are "
                    "correct; they measure different things. The model incorrectly treats them "
                    "as contradictory. The 8% attributable-risk figure is factually accurate."
                ),
            }

        return None

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def get_sliding_window_chunks(self, text: str) -> List[str]:
        size = self.cfg.get("CHUNK_SIZE", 5000)
        overlap = self.cfg.get("CHUNK_OVERLAP", 1000)
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start += max(1, size - overlap)
        return chunks

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_exhaustive_assertions(
        self, text: str, chapter_name: str
    ) -> List[Dict]:
        chunks = self.get_sliding_window_chunks(text)
        all_raw: List[Dict] = []
        for chunk in chunks:
            for _ in range(self.cfg.get("EXTRACTION_RUNS", 3)):
                p = prompt.EXTRACTION_PROMPT.format(content=chunk)
                time.sleep(self.cfg.get("EXTRACTION_WAIT", 4))  # stay under RPM limit
                resp = self.generate_with_retry(p)
                if resp and getattr(resp, "text", None):
                    parsed = self._safe_json_loads(resp.text)
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    if isinstance(parsed, list):
                        for item in parsed:
                            if not isinstance(item, dict):
                                continue
                            orig = (item.get("original_statement") or "").strip()
                            opt = (item.get("optimized_assertion") or orig).strip()
                            if orig:
                                all_raw.append(
                                    {
                                        "original_statement": orig,
                                        "optimized_assertion": opt,
                                        "chapter": chapter_name,
                                    }
                                )
                time.sleep(self.cfg.get("WAIT_TIME", 2))
        return all_raw

    def extract_assertions_multi_run(
        self, text: str, source_name: str
    ) -> List[Dict]:
        return self.extract_exhaustive_assertions(text, source_name)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def create_master_list(self, raw_list: List[Dict]) -> List[Dict]:
        seen: Dict[str, bool] = {}
        master: List[Dict] = []
        for item in raw_list or []:
            orig = (item.get("original_statement") or "").strip()
            opt = (item.get("optimized_assertion") or "").strip()
            if orig and orig not in seen:
                seen[orig] = True
                master.append(
                    {
                        "index": len(master),
                        "original_statement": orig,
                        "optimized_assertion": opt or orig,
                    }
                )
        os.makedirs("output", exist_ok=True)
        csv_path = os.path.join("output", "stage1_master_assertions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "original_statement", "optimized_assertion"])
            for a in master:
                writer.writerow(
                    [
                        a.get("index"),
                        a.get("original_statement"),
                        a.get("optimized_assertion"),
                    ]
                )
        return master

    # ------------------------------------------------------------------
    # Evidence retrieval
    # ------------------------------------------------------------------

    def search_tavily(self, query: str) -> List[Dict]:
        """
        Search Tavily for evidence. Returns list of result dicts.

        Auth: tries Authorization Bearer header (v2) first; falls back to
        api_key in body (v1) on 401.

        Rate-limit handling: retries up to 3 times with exponential backoff
        on HTTP 429 (Too Many Requests). Logs all non-200 responses so
        citation failures are visible in logs rather than silent.
        """
        if not self.tavily_key:
            return []

        import logging
        log = logging.getLogger(__name__)

        payload_v2 = {
            "query": query,
            "search_depth": "basic",  # basic=1 credit vs advanced=2; sufficient for Stage 1 pre-filter
            "max_results": 5,
        }
        headers_v2 = {
            "Authorization": f"Bearer {self.tavily_key}",
            "Content-Type": "application/json",
        }
        payload_v1 = {
            "api_key": self.tavily_key,
            "query": query,
            "search_depth": "basic",  # basic=1 credit vs advanced=2; sufficient for Stage 1 pre-filter
            "max_results": 5,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Attempt 1: v2 Bearer header
                r = requests.post(
                    "https://api.tavily.com/search",
                    json=payload_v2,
                    headers=headers_v2,
                    timeout=20,
                )

                # Fallback: v1 body key on 401
                if r.status_code == 401:
                    r = requests.post(
                        "https://api.tavily.com/search",
                        json=payload_v1,
                        timeout=20,
                    )

                if r.status_code == 429:
                    # Rate limited - wait and retry with backoff
                    wait = (attempt + 1) * 10  # 10s, 20s, 30s
                    log.warning(f"Tavily rate limit (429). Waiting {wait}s before retry {attempt+1}/{max_retries}.")
                    time.sleep(wait)
                    continue

                if r.status_code != 200:
                    log.warning(f"Tavily API error {r.status_code}: {r.text[:200]}")
                    return []

                data = r.json()
                return data.get("results", []) if isinstance(data, dict) else []

            except requests.exceptions.Timeout:
                log.warning(f"Tavily timeout (attempt {attempt+1}) for query: {query[:60]}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return []
            except Exception as e:
                log.warning(f"Tavily error: {e}")
                return []

        return []

    # ------------------------------------------------------------------
    # Stage 1 fact-checking - main entry point
    # ------------------------------------------------------------------

    def run_stage1_factcheck(self, master_list: List[Dict]) -> List[Dict]:
        """
        For each assertion:
          1. Retrieve Tavily evidence (two attempts: full claim, then 7-word query)
          2. Call Gemini with STAGE1_PROMPT
          3. Apply overrides A-E in order
          4. Return list of verdict dicts

        Override order matters:
          F (structural) before A (contradiction) before B (absolute language)
          before C (numeric) before D (hard-coded wrong) before E (rescue).
          D runs on Correct AND Flagged for Review (not Incorrect) so that
          model-self-flagged claims with known-wrong facts still get Incorrect.
          E (Incorrect->Correct/Flagged) runs last so it can also rescue cases
          where a previous override mis-fired.
        """
        final: List[Dict] = []

        for i, item in enumerate(master_list or [], 1):
            claim = (item.get("optimized_assertion") or "").strip()

            if not claim:
                final.append(
                    {
                        "index": i,
                        "original_statement": item.get("original_statement", ""),
                        "optimized_assertion": claim,
                        "final_verdict": "Flagged for Review",
                        "reasoning": "Empty assertion after optimization; cannot verify.",
                        "citations": [],
                    }
                )
                continue

            # --- Evidence retrieval (two attempts) ---
            evidence = self.search_tavily(claim)
            if not evidence:
                short_query = " ".join(claim.split()[:7])
                evidence = self.search_tavily(short_query)

            # extract URLs from Tavily results for citations export
            tavily_urls = self._dedupe_preserve_order(
                [
                    e.get("url", "")
                    for e in (evidence or [])
                    if isinstance(e, dict) and isinstance(e.get("url", None), str) and e.get("url", "").strip()
                ]
            )

            evidence_str = (
                "\n".join(
                    f"- {e.get('content', '')} ({e.get('url', '')})"
                    for e in evidence
                    if isinstance(e, dict)
                )
                if evidence
                else "No web evidence retrieved. Use your medical knowledge."
            )

            # --- Model call ---
            p = self.STAGE1_PROMPT.format(assertion=claim, evidence=evidence_str)
            resp = self.generate_with_retry(p)
            verdict = self._safe_json_loads(resp.text) if resp else {}

            if isinstance(verdict, list) and verdict:
                verdict = verdict[0]

            # Guard: empty or invalid model output
            if not isinstance(verdict, dict) or not verdict.get("final_verdict"):
                verdict = {
                    "final_verdict": "Flagged for Review",
                    "reasoning": "Invalid or empty model output; flagged for manual review.",
                    "analysis": "",
                    "concerns": [],
                }

            # --- Extract new chain-of-thought fields ---
            # Model now returns analysis + concerns + final_verdict + confidence + reasoning.
            # Merge analysis into reasoning for output if reasoning is absent.
            if not verdict.get("reasoning") and verdict.get("analysis"):
                verdict["reasoning"] = verdict["analysis"]

            # --- Concerns gate: if model listed ANY concern, downgrade to Flagged ---
            # This implements the principle: any voiced doubt = Flagged.
            # The model is instructed to list every concern, so a non-empty concerns
            # array is an explicit signal that something needs Stage 2 verification.
            concerns = verdict.get("concerns", [])
            if isinstance(concerns, list) and len(concerns) > 0:
                if verdict.get("final_verdict") in ("Correct", "Incorrect"):
                    original_verdict = verdict["final_verdict"]
                    verdict["final_verdict"] = "Flagged for Review"
                    verdict["reasoning"] = (
                        f"[Concerns raised by model] Flagged because model identified: "
                        f"{'; '.join(str(c) for c in concerns[:3])}. "
                        f"Original verdict was {original_verdict}. "
                        + verdict.get("reasoning", "")
                    )

            # --- Confidence gate: auto-downgrade low-confidence Correct/Incorrect ---
            # Threshold raised from 0.95 -> 0.97 (Session 13 redesign).
            # Evidence: all false-Correct/Incorrect errors across sessions 1-12 had
            # confidence ≥ 0.95 but were wrong. Tighter threshold reduces the window
            # for confidently-wrong model calls to survive as Correct/Incorrect.
            confidence = verdict.get("confidence", 1.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 1.0
            if verdict.get("final_verdict") in ("Correct", "Incorrect") and confidence < 0.97:
                original_verdict = verdict["final_verdict"]
                verdict["final_verdict"] = "Flagged for Review"
                verdict["reasoning"] = (
                    f"[Low confidence: {confidence:.2f}] Model was not sufficiently certain "
                    f"to confirm as {original_verdict}. "
                    + verdict.get("reasoning", "")
                )

            # --- Override F: structural high-risk pattern detector ---
            # Fires BEFORE all other overrides. Auto-flags structurally risky claims
            # regardless of model verdict or confidence, covering the 6 error categories
            # observed across all production sessions. This is proactive, not reactive.
            # Applies to both Correct AND Incorrect verdicts.
            if verdict.get("final_verdict") in ("Correct", "Incorrect"):
                structural_flag = self._structural_high_risk_check(claim)
                if structural_flag:
                    verdict["final_verdict"] = "Flagged for Review"
                    verdict["reasoning"] = structural_flag

            # --- Override A: self-contradiction guard ---
            # Now checks BOTH reasoning AND analysis fields (new CoT output).
            # Fires when model said Correct but its own reasoning reveals an error/stale fact.
            combined_reasoning = (
                verdict.get("reasoning", "") + " " + verdict.get("analysis", "")
            ).strip()
            if self._verdict_contradicts_reasoning(
                verdict.get("final_verdict", ""),
                combined_reasoning,
                claim,
            ):
                verdict["final_verdict"] = "Flagged for Review"
                verdict["reasoning"] = (
                    "[Auto-flagged] Model reasoning contradicted Correct verdict: "
                    + verdict.get("reasoning", "")
                )

            # --- Override B: absolute-language veto ---
            if (
                verdict.get("final_verdict") == "Correct"
                and self._contains_absolute_language(claim)
            ):
                verdict["final_verdict"] = "Flagged for Review"
                verdict["reasoning"] = (
                    "Assertion contains absolute language (always/exclusively/"
                    "universally/only X) that cannot be certified without "
                    "systematic evidence."
                )

            # --- Override C: numeric self-check ---
            if (
                verdict.get("final_verdict") == "Correct"
                and self._has_specific_numbers(claim)
            ):
                check = self._numeric_self_check(claim)
                if not check.get("accurate", True):
                    verdict["final_verdict"] = "Flagged for Review"
                    verdict["reasoning"] = (
                        f"[Numeric check] {check.get('issue', 'Statistic may be inaccurate.')} "
                        f"Original reasoning: {verdict.get('reasoning', '')}"
                    )

            # --- Override D: hard-coded false-Correct catches ---
            # SESSION 14 BUG 1 FIX: condition changed from == "Correct" to
            # in ("Correct", "Flagged for Review") so that claims the model self-flags
            # via Mandatory Flag rules (Types A-G) are still checked against confirmed-wrong
            # patterns. Deliberately excludes "Incorrect" to prevent P7/P8 (which return
            # Flagged for Review) from downgrading confirmed-wrong Incorrect verdicts.
            if verdict.get("final_verdict") in ("Correct", "Flagged for Review"):
                hard_check = self._hard_coded_fact_check(claim)
                if hard_check:
                    verdict["final_verdict"] = hard_check["verdict"]
                    verdict["reasoning"] = hard_check["reasoning"]
                    verdict["analysis"] = hard_check["reasoning"]  # sync: override replaces stale model analysis
                    verdict["concerns"] = []  # clear: stale model concerns no longer apply
                    verdict["analysis"] = hard_check["reasoning"]  # sync: override replaces stale model analysis
                    verdict["concerns"] = []  # clear: stale model concerns no longer apply

            # --- Override E: hard-coded false-Incorrect rescue ---
            # Runs last - can rescue from model verdict AND from any over-fired override.
            if verdict.get("final_verdict") == "Incorrect":
                rescue = self._hard_coded_rescue(claim)
                if rescue:
                    verdict["final_verdict"] = rescue["verdict"]
                    verdict["reasoning"] = rescue["reasoning"]
                    verdict["analysis"] = rescue["reasoning"]  # sync: override replaces stale model analysis
                    verdict["concerns"] = []  # clear: stale model concerns no longer apply
                    verdict["analysis"] = rescue["reasoning"]  # sync: override replaces stale model analysis
                    verdict["concerns"] = []  # clear: stale model concerns no longer apply

            # finalize citations (use model citations if present, else Tavily URLs)
            model_citations = verdict.get("citations", [])
            if not isinstance(model_citations, list):
                model_citations = []
            model_citations = self._dedupe_preserve_order(
                [c for c in model_citations if isinstance(c, str)]
            )

            final_citations = model_citations if model_citations else tavily_urls

            # --- Finalise row ---
            verdict.update(
                {
                    "index": i,
                    "original_statement": item["original_statement"],
                    "optimized_assertion": claim,
                    "citations": final_citations,
                }
            )

            final.append(verdict)
            time.sleep(self.cfg.get("WAIT_TIME", 2))

        return final