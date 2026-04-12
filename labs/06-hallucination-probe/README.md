# Lab 06: The Hallucination Probe

**Act:** V — Where They Break
**Time:** 60–90 minutes
**Hardware:** CPU only (uses API calls, no local model needed)
**Requirements:** An OpenRouter API key (set as `OPENROUTER_API_KEY` env var)

## Learning Objectives

- Design a factual question bank split into "common knowledge" and "long-tail" categories
- Query small language models via the OpenRouter API and collect responses
- Score responses with deterministic multi-alias matching
- Compute hallucination rates per category and compare to the Kalai-Vempala bound
- Understand why hallucination is a mathematical inevitability, not a bug to fix

## Prerequisites

- None — this lab uses API calls, not local model internals

## Setup

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Artifact

A scored question bank and a hallucination-rate chart showing the frequency-dependent
bound on confident errors.

## Run

```bash
just run 06
# or: python labs/06-hallucination-probe/lab.py
```
