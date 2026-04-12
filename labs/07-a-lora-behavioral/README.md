# Lab 07-A: LoRA for Behavioral Fine-Tuning

**Act:** VI — Making It Yours
**Time:** 60–90 minutes
**Hardware:** GPU (CUDA or MPS) with 4+ GB memory. CPU works but slow.

## Learning Objectives

- Implement LoRA from scratch — the A×B low-rank decomposition, scaling, and zero-init
- Attach adapters to Qwen3-0.6B and fine-tune on 20 cooking instruction examples
- See how 24,576 trainable parameters can shift the model's response style
- Compare before/after responses to see fine-tuning in action
- Merge the adapter back into base weights and verify identical output
- Save a few-MB adapter file instead of a full model copy

## Prerequisites

- Lab 03 (Build a Transformer Block) — understanding of projection layers
- Basic familiarity with PyTorch training loops

## What This Lab Covers

This is the first of two LoRA labs:
- **07-A (this lab)**: Behavioral fine-tuning — teaching the model a domain's
  response style through natural-language instruction examples
- **07-B**: Tool-calling fine-tuning — teaching the model to emit structured
  JSON for function invocation

## Artifact

A trained LoRA adapter file (~2 MB) fine-tuned for cooking Q&A.

## Run

```bash
just run 07-a
# or: python labs/07-a-lora-behavioral/lab.py
```
