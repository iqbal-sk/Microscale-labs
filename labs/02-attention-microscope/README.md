# Lab 02: Attention Under the Microscope

**Act:** II — Inside the Machine
**Time:** 60–90 minutes
**Hardware:** CPU or Apple Silicon (model is ~1.2 GB). GPU optional but faster.

## Learning Objectives

- Load a real language model and extract raw attention weight matrices
- Visualize attention patterns across all 448 heads (28 layers x 16 heads)
- Identify three known pattern types: sink heads, previous-token heads, and positional heads
- Understand Grouped Query Attention (GQA) by observing paired heads
- Ablate individual heads and measure the perplexity impact
- Build intuition for "some heads matter, most don't"

## Prerequisites

- Lab 01 (The Token Tax) — familiarity with tokenizers and the microscale package
- Basic understanding of attention (what Q, K, V mean conceptually)

## Model

Qwen3-0.6B (~1.2 GB download). Uses Grouped Query Attention: 16 query heads, 8 key-value heads.

## Artifact

- A gallery of attention head heatmaps with pattern annotations
- An ablation impact heatmap (28x16 grid showing which heads matter)

## Run

```bash
# As a script
just run 02

# Or directly
python labs/02-attention-microscope/lab.py

# As a notebook
jupyter lab labs/02-attention-microscope/lab.py
```
