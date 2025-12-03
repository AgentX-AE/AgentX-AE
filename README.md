# AgentX Simulator  
*(Artifact for ISCA 2026 Review Only)*

This repository contains the simulation code used in our ISCA 2026 submission on **using Hybrid Bonding (HB) + Near Data Processing (NDP)** to accelerate **LLM Agent** workloads.  
The code is a **research prototype** built on top of **Ramulator 2.0**, with custom support for **AgentX**.

> ⚠️ **Usage & Distribution**
> - This repository is provided **only for anonymous ISCA 2026 artifact evaluation**.
> - It is intended **solely for reviewers of the corresponding paper**.
> - Please **do not redistribute, fork, or publicly mirror** this code or derivative works.

---

## 1. Project Overview

- We model a **GPU / AgentX  system** for LLM Agent inference.
- The simulator focuses on **NDP-style operations**, capturing:
  - LPDDR5 timing and command behavior,
  - and NDP command scheduling / trace flows.

The top-level structure of this repository follows Ramulator 2.0’s layout, with our AgentX-specific modifications confined to the src directory and applied to a vanilla Ramulator 2.0 checkout via the set_AgentX.sh script.

---

## 2. Dependencies

This artifact assumes the following software stack:

- **g++**: 12.3.0  
- **CMake**: ≥ 3.14  
- **Python**: 3.10.19  
- **vLLM**: 0.11 (installed in the Python environment used for real-latency tests)  
- At least one **NVIDIA H100 PCIe 80GB GPU** for real latency measurements.

---

## 3. Build & Run Instructions

### 3.1. Clone the Github repository

```bash
$ git clone https://github.com/AgentX-AE/AgentX-AE.git
$ cd AgentX-AE
$ git submodule update --init --recursive
$ cd AgentX
```

### 3.2 Build the AgentX Simulator (Modified Ramulator2 Backend)

From the AgentX directory, run:

```bash
$ bash set_AgentX.sh
$ cd ramulator2
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ cp AgentX ../../
$ cd ../../
```

### 3.3 Collect Real H100 Latency with vLLM

We first obtain **real LLM inference latencies** on an H100 PCIe 80GB GPU using vLLM.

From the AgentX directory, run:

```bash
$ cd src
$ bash run_real_latency.sh qwen3-32b /data1/models/qwen3-32b 32B 0
$ bash run_real_latency.sh qwen3-14b /data1/models/qwen3-14b 14B 0
$ bash run_real_latency.sh qwen3-8b  /data1/models/qwen3-8b   8B  0
$ cd ..
```
- first argument is the model name used in our configs (e.g., qwen3-32b).
- second argument is the absolute path to the model weights on your machine.
- third argument denotes the model scale (e.g., 8B, 14B, 32B).
- fourth argument is the GPU index (e.g., 0 for the first H100).
- To run the simulations, you must provide latency measurements for at least three model sizes: **8B, 14B, and 32B**.

### 3.4 Run AgentX Latency Simulation

After collecting real H100 latencies and building the AgentX backend, we run the AgentX simulations.

From the AgentX directory, run:

```bash
$ python main.py --dataset DATASET_NAME
```

**DATASET_NAME** should follow the dataset naming used in the paper (i.e., the LLM Dataset evaluated in the ISCA submission).

### 3.5 Run AgentX Directly

After building the executable (Section 4.1) and generating the corresponding LLM inference traces by invoking gen_trace.py in src/, you can directly launch AgentX with its default configuration.

From the AgentX directory, run:

```bash
$ ./AgentX -f AgentX.yaml
```
