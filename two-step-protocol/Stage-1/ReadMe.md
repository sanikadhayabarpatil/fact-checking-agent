# Stage 1 – Initial Screening (High Sensitivity)

Stage 1 is the **first pass** of the Two-Step Fact-Checking Protocol.

Its goal is **not** to fully verify claims, but to:
- Extract all factual assertions
- Attach available evidence
- Identify which claims require deeper review

---

## 🎯 Objectives

- Maximize recall of factual claims
- Prevent hallucinated correctness
- Surface uncertainty explicitly
- Produce structured, auditable outputs

---

## 🔄 Stage 1 Workflow

### **Step 1: Assertion Extraction (Gemini ×3)**

- The chapter is processed **three separate times**
- Each run extracts factual, testable assertions
- Outputs are merged to maximize recall

Why 3 runs?
- LLM extraction is non-deterministic
- Multiple passes reduce missed claims

---

### **Step 2: Deduplication**

- Near-duplicate assertions are normalized and merged
- Produces a clean **master assertions list**

---

### **Step 3: Evidence Gathering (Tavily ×1)**

- Each master assertion is queried once using Tavily
- Results are optionally domain-restricted (e.g., PubMed / NCBI)
- Retrieved URLs and snippets are attached verbatim

Important:
- Tavily failing to return evidence ≠ claim is false
- It only means the claim could not be verified under Stage 1 constraints

---

### **Step 4: Initial Fact-Checking (Gemini ×1, Batched)**

Gemini evaluates each assertion using:
- Tavily evidence (primary)
- Chapter excerpts (context only)
- General scientific reasoning

---

## ✅ Verdict Rules (Critical)

Each assertion receives **exactly one** verdict:

### **Correct**
ONLY if:
- At least **one external citation URL** is provided
- The citation directly supports the claim

### **Incorrect**
ONLY if:
- External evidence clearly contradicts the claim

### **Flagged for Review**
Used whenever:
- No external citation is available
- Evidence is weak, indirect, or ambiguous
- Claim is numerical, historical, or attribution-sensitive
- Verification relies mainly on the chapter text
- There is any uncertainty

> **When in doubt → Flagged for Review**

---

## 🧪 Why So Many Flags?

This is intentional.

Stage 1 prioritizes **sensitivity over precision**:
- It is better to flag a true claim than to wrongly mark a false one as correct
- Flagged assertions are resolved in Stage 2

---

## 📂 Outputs

Stage 1 produces:

- `stage1_master_assertions.json`
- `stage1_initial_results.json`
- `stage1_flagged_assertions.json`
- `stage1_metadata.json`

All outputs are:
- Structured
- Auditable
- Free of hallucinated citations

---

## ✅ Current Behavior (Expected)

- High flag rate (~40–60%)
- Fewer but well-supported “Correct” claims
- Zero silent failures
- Zero citation-less “Correct” verdicts

---

## 🔜 Next Step

All `Flagged for Review` assertions flow into **Stage 2**, where deeper, multi-pass verification is performed.

Stage 1 is complete and stable.

