# Lab 01: The Token Tax

**Act:** I — The Landscape
**Time:** 30 minutes
**Hardware:** CPU only (runs in 10 seconds)

## Learning Objectives

- Load and compare 4 BPE tokenizers (o200k, cl100k, p50k, gpt2)
- Measure the "token tax" across 5 languages
- Produce a heatmap showing tokenizer fairness gaps
- Understand why tokenizer vocabulary size matters for multilingual equity

## Prerequisites

- None — this is the first lab

## Artifact

A heatmap (PNG + interactive HTML) comparing token efficiency across languages and tokenizers.

## Run

**As a script:**
```bash
just run 01
# or: python labs/01-token-tax/lab.py
```

**As a notebook:**
```bash
jupyter lab labs/01-token-tax/lab.py
```
