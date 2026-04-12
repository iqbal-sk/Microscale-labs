# Lab 10: The Roofline Lab

**Act:** VIII — Serving the Model
**Time:** 60–90 minutes
**Hardware:** Any GPU (CUDA or MPS). CPU-only will not produce meaningful results.

## Learning Objectives

- Measure your GPU's actual memory bandwidth and compute throughput
- Build a roofline chart from YOUR hardware's real numbers
- Plot a model's arithmetic intensity at different batch sizes
- See why LLM decode is bandwidth-bound and prefill becomes compute-bound
- Understand the fundamental bottleneck of transformer inference

## Prerequisites

- Lab 03 (Build a Transformer Block) — understanding of model forward pass
- Basic understanding of GPU concepts (memory, compute)

## Artifact

A roofline chart with your GPU's measured ridge and your model's operating points.

## Run

```bash
just run 10
# or: python labs/10-roofline-lab/lab.py
```
