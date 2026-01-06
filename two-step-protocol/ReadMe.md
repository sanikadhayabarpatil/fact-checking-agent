# Two-Step Fact-Checking Protocol

This repository implements a **Two-Step Fact-Checking Protocol** designed to verify scientific and biomedical text with **high reliability and minimal hallucination**.

The protocol separates **high-sensitivity screening** from **high-precision verification**, ensuring that claims are never prematurely marked as correct without sufficient evidence.

---

## 🎯 Design Goals

- Avoid hallucinated confirmations
- Make uncertainty explicit
- Separate screening from deep verification
- Produce outputs that are auditable and review-ready
- Scale to long scientific chapters

---

## 🧠 Core Philosophy

> **Plausible ≠ Verified**

A claim should only be marked *Correct* when it is explicitly supported by external evidence.  
If evidence is missing, weak, or ambiguous, the system must escalate rather than guess.

---

## 🔁 Protocol Overview

### **Stage 1: Initial Screening (High Sensitivity)**

Purpose:
- Capture *all* factual assertions
- Attach available evidence
- Escalate uncertainty conservatively

Key characteristics:
- Maximizes recall
- Does NOT attempt deep verification
- Flags anything that cannot be confidently supported

Outputs:
- Correct
- Incorrect
- Flagged for Review

---

### **Stage 2: Deep Verification (High Precision)**

Purpose:
- Resolve flagged assertions from Stage 1
- Perform multi-hop, citation-level verification
- Produce final, publication-grade judgments

Key characteristics:
- Fewer assertions
- More intensive reasoning and retrieval
- Higher computational cost, higher confidence

---

## 🔒 Reliability Guarantees

The protocol enforces:
- No “Correct” verdict without external citations
- No silent failures (empty outputs are treated as errors)
- Clear traceability from claim → evidence → verdict

---

## 📌 Why Two Stages?

Single-pass fact checking often:
- Over-trusts the source text
- Hallucinates support
- Collapses uncertainty

By splitting the process:
- Stage 1 surfaces uncertainty early
- Stage 2 focuses effort where it matters

---

## ✅ Current Status

- Stage 1: Implemented and stable
- Stage 2: Planned / in progress

See `README_STAGE1.md` for details on Stage 1.
