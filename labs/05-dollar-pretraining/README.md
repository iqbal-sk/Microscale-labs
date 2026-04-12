# Lab 05: The $1 Pretraining Run

**Act:** IV — How They Learn
**Time:** 90–120 minutes (including ~20 min of actual training)
**Hardware:** Any GPU (CUDA/MPS) recommended. CPU works but slower.

## Learning Objectives

- Build a 10M-parameter GPT-2 model from scratch and understand its parameter budget
- Train it on TinyStories — real synthetic data used in language model research
- Watch the loss curve descend from random noise to coherent language
- Compare training on clean vs corrupted data to see the textbook hypothesis in action
- Generate text from your trained model and evaluate quality

## Prerequisites

- Lab 03 (Build a Transformer Block) — understanding of transformer architecture
- Basic familiarity with training loops (optimizer, loss, backpropagation)

## Artifact

Two trained 10M-parameter models, their loss curves, and a side-by-side
comparison of generation quality.

## Run

```bash
just run 05
# or: python labs/05-dollar-pretraining/lab.py
```
