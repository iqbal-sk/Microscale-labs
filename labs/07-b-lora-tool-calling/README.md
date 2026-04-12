# Lab 07-B: LoRA for Tool Calling

**Act:** VI — Making It Yours
**Time:** 60–90 minutes
**Hardware:** GPU (CUDA or MPS) with 4+ GB memory. CPU works but slow.

## Learning Objectives

- Apply LoRA to teach a model **structured output** (JSON tool calls)
- Understand why tool-calling needs **higher-rank adapters** and **more target modules**
  than plain instruction following
- Format training examples with Qwen3's native tool-calling template
- Measure success with a deterministic evaluator: parse the JSON, verify
  the function name and arguments
- Generalize to held-out prompts not seen during training

## Prerequisites

- Lab 07-A (LoRA for Behavioral Fine-Tuning) — understanding the LoRA implementation

## What This Lab Covers

This is the second of two LoRA labs:
- **07-A**: Behavioral fine-tuning — natural-language response style
- **07-B (this lab)**: Tool-calling fine-tuning — structured JSON function calls

The key difference: tool-calling requires the model to generate **exact syntax**
(function names, argument names, valid JSON). This is a harder task than
free-form instruction following, so we use **higher rank (r=16)** and
target both `q_proj` and `v_proj` projections.

## Artifact

A LoRA adapter that produces valid JSON tool calls for the 6-function
kitchen assistant API.

## Run

```bash
just run 07-b
# or: python labs/07-b-lora-tool-calling/lab.py
```
