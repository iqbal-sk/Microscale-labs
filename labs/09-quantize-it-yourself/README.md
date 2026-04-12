# Lab 09: Quantize It Yourself

**Act:** VII — Packing for Travel
**Time:** 90 minutes
**Hardware:** CPU only. Single tensor fits in RAM.

## Learning Objectives

- Implement three quantization schemes from scratch: naive 4-bit, NF4, and K-quant
- Understand why NF4 uses Gaussian quantiles and why K-quant uses sub-block scales
- Measure quantization error (MSE, SQNR) for each method
- See the weight distribution that makes NF4 work — and why naive binning wastes precision
- Connect quantization quality to model perplexity

## Prerequisites

- Lab 03 (Build a Transformer Block) — understanding of weight matrices
- Lab 04 (Model Autopsy) — familiarity with model weight shapes

## Artifact

Three quantized tensor files, an error comparison chart, and a reference implementation.

## Run

```bash
just run 09
# or: python labs/09-quantize-it-yourself/lab.py
```
