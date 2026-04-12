# Lab 11: KV Cache Budget Calculator

**Act:** VIII — Serving the Model
**Time:** 60 minutes
**Hardware:** CPU only for the calculator. GPU optional for live measurement.

## Learning Objectives

- Understand why the KV cache exists and why it grows with sequence length
- Build a calculator that predicts exact KV cache memory from model config
- Compare predicted vs actual memory consumption on a real model
- See how batch size, sequence length, and quantization trade off against memory
- Learn why KV cache is the limiting factor for concurrent users

## Prerequisites

- Lab 03 (Build a Transformer Block) — understanding of K, V projections
- Lab 10 (Roofline Lab) — understanding of memory constraints

## Artifact

A KV cache calculator and memory budget charts for multiple models.

## Run

```bash
just run 11
# or: python labs/11-kv-cache-calculator/lab.py
```
