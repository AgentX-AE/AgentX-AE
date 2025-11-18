# AgentX-AE: Architectural Evaluation Framework for Agent-Centric LLM Workloads

This repository contains the reference implementation, simulation scripts, and experimental artifacts for our ISCA 2026 submission **AgentX-AE**.  
The project is designed to evaluate next-generation agent-centric large language model (LLM) workloads, focusing on memory-centric bottlenecks, heterogeneous model pools, and near-memory/near-storage acceleration strategies.

## 🎯 Project Purpose

This codebase accompanies our **ISCA 2026** paper submission.  
It provides:

- A unified agent-level workload generator for multi-model pipelines.
- End-to-end performance measurement for prefill/decode phases.
- Simulation of heterogeneous model pools approaching **1 TB+** capacity.
- Architectural evaluation of PIM/NDP-enabled memory hierarchies.
- Comparisons against GPU baselines (H100, multi-GPU, CPU-offload, etc.).
- Reproducible scripts for all figures and tables in the paper.

The code is released to support artifact evaluation (AE) and enable reproducibility.

## 📂 Repository Structure

