# Lab 08: Your First DPO Alignment

**Act:** VI — Making It Yours
**Time:** 90 minutes
**Hardware:** GPU (CUDA or MPS) with 8+ GB memory.

## Learning Objectives

- Understand the DPO loss function and how it differs from SFT
- Build preference data: what makes a "chosen" response better than "rejected"
- Train with TRL's DPOTrainer using LoRA adapters
- Compare model behavior before and after alignment
- See how 20 preference pairs can shift a model's response style

## Prerequisites

- Lab 07 (LoRA in 50 Lines) — understanding of LoRA adapters

## Artifact

A DPO-aligned LoRA adapter and a before/after comparison report.

## Run

```bash
just run 08
# or: python labs/08-dpo-alignment/lab.py
```
