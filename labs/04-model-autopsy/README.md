# Lab 04: Model Autopsy

**Act:** III — The Current Champions
**Time:** 45–60 minutes
**Hardware:** CPU only. Downloads only headers (~35KB per model), not weights.

## Learning Objectives

- Parse the safetensors binary format to read model metadata without loading weights
- Detect architecture choices (GQA ratio, tied embeddings, FFN type) from tensor names alone
- Compare 4 SLMs (360M to 3.8B) side by side: parameter distribution, design trade-offs
- See how the vocabulary tax shrinks as models grow larger

## Prerequisites

- Lab 03 (Build a Transformer Block) — understanding of Q/K/V projections, FFN structure

## Artifact

A comparison table and visualizations comparing SmolLM2-360M, Qwen3-0.6B, SmolLM3-3B, and Phi-4-mini.

## Run

```bash
just run 04
# or: python labs/04-model-autopsy/lab.py
```
