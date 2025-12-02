import argparse
import math
import os
import pandas
from src.model_config import *
from src.agent_config import *
import subprocess
from pathlib import Path


def run_lpddrpim(
    modelsize: str, context_len: int, batch_size: int = 1,
    maxlen: int = 32768, dbyte: int = 2, output: str = "AgentX-NDP.trace",
    agentx_dir: str = ".", yaml_file: str = "AgentX.yaml"
) -> int:
    
    agentx_path = Path(agentx_dir)
    gen_trace_py = agentx_path / "src" / "gen_trace.py"
    if not gen_trace_py.exists():
        raise FileNotFoundError(f"Cannot find {gen_trace_py}. Please check your AgentX directory structure.")

    gen_trace_cmd = ["python", str(gen_trace_py),
                     "-modelsize", modelsize,
                     "-len", str(context_len),
                     "-batch", str(batch_size),
                     "-maxl", str(maxlen),
                     "-db", str(dbyte),
                     "-o", output]

    try:
        subprocess.run(gen_trace_cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,text=True,
                       check=True,cwd=agentx_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run gen_trace.py.\n"
                           f"Command: {' '.join(gen_trace_cmd)}\n"
                           f"Output:\n{e.stdout}")

    agentx_bin = agentx_path / "AgentX"
    if not agentx_bin.exists():
        raise FileNotFoundError(f"Cannot find AgentX binary at {agentx_bin}. Please build AgentX first.")

    run_cmd = ["./AgentX", "-f", yaml_file]

    try:
        run_res = subprocess.run(run_cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 check=True, cwd=agentx_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run AgentX.\n"
                           f"Command: {' '.join(run_cmd)}\n"
                           f"Output:\n{e.stdout}")

    cycles = None
    for line in run_res.stdout.splitlines():
        if "memory_system_cycles" in line:
            parts = line.split()
            try:
                cycles = int(parts[-1])
            except (ValueError, IndexError):
                pass

    if cycles is None:
        raise RuntimeError("Cannot find 'memory_system_cycles' in AgentX output.\n"f"Full output:\n{run_res.stdout}")

    rm_trace_cmd = f"rm {output}"
    
    try:
        os.system(rm_trace_cmd)
    except Exception as e:
        print(f"Error: {e}")
    
    rm_log_cmd = f"rm -rf log"
    
    try:
        os.system(rm_log_cmd)
    except Exception as e:
        print(f"Error: {e}")
    
    return cycles


def main():
    # clcyes = run_lpddrpim(
    #     modelsize="32B",
    #     context_len=8192,
    #     batch_size=1,
    #     maxlen=32768,
    #     dbyte=2,
    #     output="LPDDRPIM.trace",
    #     agentx_dir=".",
    #     yaml_file="LPDDRPIM.yaml",
    # )
     
    parser = argparse.ArgumentParser(
        description="Model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--dataset",
                        type=str,
                        default='BBH',
                        help="dataset name. default=BBH")
    parser.add_argument("--batchsize",
                        type=int,
                        default=1,
                        help="batch size. default=1")
    parser.add_argument("--maxlen",
                        type=int,
                        default=32768,
                        help="maximum context length. default=32768")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB). default=80")
    parser.add_argument("--device",
                        type=str,
                        default='H100 and AgentX',
                        help="device type. H100 or AgentX")
    parser.add_argument("--dtype",
                        type=int,
                        default=2,
                        help="data type (B). default=2")

    args = parser.parse_args()
    dataset = args.dataset
    batch_size = args.batchsize
    maxlen = args.maxlen
    dtype = args.dtype
    if args.device == "H100 and AgentX":
        H100_time = get_prefill_time(dataset, "H100") + \
                    get_pcle_time(dataset, "H100") + \
                    get_decode_time(dataset, "H100")
        print("Total latency on H100 for", dataset, ":", H100_time, "s")
        AgentX_time = get_prefill_time(dataset, "AgentX") + \
                      get_AgentX_time(dataset, "AgentX", batch_size, maxlen, dtype, run_lpddrpim)
        print("Total latency on AgentX for", dataset, ":", AgentX_time, "s")
        print("Speedup (H100 / AgentX):", H100_time / AgentX_time)
    else:
        raise ValueError(f"Unknown device type: {args.device}")

if __name__ == "__main__":
    main()
