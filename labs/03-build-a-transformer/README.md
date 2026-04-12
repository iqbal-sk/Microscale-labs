# Lab 03: Build a Transformer Block from Raw Ops

**Act:** II — Inside the Machine
**Time:** 90–120 minutes (the hardest lab)
**Hardware:** CPU or Apple Silicon. ~1.2 GB model download.

## Learning Objectives

- Implement RMSNorm, Rotary Position Embeddings, Grouped Query Attention, and SwiGLU from scratch
- Load real weights from Qwen3-0.6B layer 0 into your implementation
- Verify your output matches HuggingFace's output to floating-point precision
- Understand every component of a modern transformer block — not from diagrams, but from code you wrote

## Prerequisites

- Lab 02 (Attention Under the Microscope) — understanding of attention patterns and GQA
- Comfort with PyTorch tensor operations (matmul, reshape, softmax)

## The Aha Moment

The moment `torch.allclose(yours, theirs, atol=1e-5)` returns `True` and you realize you just reproduced a production transformer layer from scratch — not a toy, the real thing with real weights.

## Artifact

A standalone `transformer_block.py` module (inside the microscale package) that loads any Qwen3 layer and produces identical outputs.

## Run

```bash
just run 03
# or: python labs/03-build-a-transformer/lab.py
```
