# Lab 07: LoRA in 50 Lines

**Act:** VI — Making It Yours
**Time:** 60–90 minutes
**Hardware:** GPU (CUDA or MPS) with 4+ GB memory. CPU works but slow.

## Learning Objectives

- Implement LoRA from scratch — the A×B low-rank decomposition, scaling, and zero-init
- Attach adapters to a real model and fine-tune on two tasks: cooking instructions and tool-calling
- Measure perplexity before and after to quantify the effect
- Merge the adapter back into base weights and verify identical output
- Save a few-MB adapter file instead of a full model copy

## Prerequisites

- Lab 03 (Build a Transformer Block) — understanding of projection layers
- Basic familiarity with PyTorch training loops

## Artifact

A trained LoRA adapter file (~2 MB) and the merged model weights.

## Run

```bash
just run 07
# or: python labs/07-lora-from-scratch/lab.py
```
