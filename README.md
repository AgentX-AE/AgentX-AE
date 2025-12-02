# AgentX-AE Flash-LPDDR-PND Simulator  
*(Artifact for ISCA 2026 Review Only)*

This repository contains the simulation code used in our ISCA 2026 submission on **using Hybrid Bonding (HB) + Processing Near Data (PND)** to accelerate **LLM Agent** workloads.  
The code is a **research prototype** built on top of **Ramulator 2.0**, with custom support for **LPDDR5 + HB + PND**.

> ⚠️ **Usage & Distribution**
> - This repository is provided **only for anonymous ISCA 2026 artifact evaluation**.
> - It is intended **solely for reviewers of the corresponding paper**.
> - Please **do not redistribute, fork, or publicly mirror** this code or derivative works.

---

## 1. Project Overview

- We model a **GPU + Flash-LPDDR5 near-data system** for LLM Agent inference.
- The simulator focuses on **PND-style operations near LPDDR5 memory**, capturing:
  - LPDDR5 timing and command behavior,
  - and PND command scheduling / trace flows.

The top-level structure of this repository follows Ramulator 2.0’s layout, with our modifications confined mainly to the `LPDDRPND` directory.

---

## 2. `LPDDRPND` Directory (Ramulator2 + PND Extensions)

The `LPDDRPND` folder is a **modified version of Ramulator 2.0**.  
On top of the original Ramulator2 codebase, we add a set of PND–specific files and modify related components to support HB + PND simulation.

Newly added core source files include:

- `lpddr_pnd_linear_mappers.cpp`  
- `LPDDR5_PND.cpp`  
- `lpddr_pnd_controller.cpp`  
- `pnd_scheduler.cpp`  
- `lpddr_all_bank_refresh.cpp`  
- `lpddr_trace_recorder.cpp`  
- `pnd_loadstore_trace.cpp`  
- `PND_LPDDR_system.cpp`  

In addition to these files, several **original Ramulator2 sources are modified** (e.g., configuration, DRAM presets, command handling) so that **PND with HB** becomes a supported configuration within the simulator.

---

## 3. Dependencies

This artifact assumes the following software stack:

- **g++**: 12.3.0  
- **CMake**: ≥ 3.14  
- **Python**: 3.10.19  
- **vLLM**: 0.11 (installed in the Python environment used for real-latency tests)  
- At least one **NVIDIA H100 PCIe 80GB GPU** for real latency measurements.

---

## 4. Build & Run Instructions

### 4.1 Build the AgentX Simulator (Modified Ramulator2 Backend)

From the project root:

```bash
$ cd AgentX
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ cp AgentX ../AgentX
$ cd ..
```

### 4.2 Collect Real H100 Latency with vLLM

We first obtain **real LLM inference latencies** on an H100 PCIe 80GB GPU using vLLM.

From the project root:

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

### 4.3 Run AgentX Latency Simulation

After collecting real H100 latencies and building the AgentX backend, we run the AgentX simulations.

From the project root:

```bash
$ python main.py --dataset DATASET_NAME
```

**DATASET_NAME** should follow the dataset naming used in the paper (i.e., the LLM Dataset evaluated in the ISCA submission).

### 4.4 Run AgentX Directly

After building the executable (Section 4.1), you can directly launch **AgentX** with its default configuration.

From the project root:

```bash
$ ./AgentX -f LPDDRPND.yaml
```
