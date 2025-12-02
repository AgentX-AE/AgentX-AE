from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import pandas as pd
from src.model_config import model_config
@dataclass
class AgentConfig:
    size: int
    cycle: float
    prefill: float
    decode: float

    def __getitem__(self, key: str):
        return getattr(self, key)


class AgentConfigStore:
    def __init__(self) -> None:
        self.length_map: Dict[float, int] = {
            0.5: 0, 1: 1, 2: 2, 3: 3, 4: 4,
            6: 5, 8: 6, 12: 7, 16: 8, 24: 9,
        }
        
        self.pcle = 32 #pcle4 x16

        self.agent_config: Dict[str, Dict[str, AgentConfig]] = {
            "LooGLE": {
                "planner": AgentConfig(size=32, cycle=1.2,  prefill=4,  decode=121),
                "critic":  AgentConfig(size=32, cycle=1.03, prefill=12, decode=179),
                "tool_s":  AgentConfig(size=8,  cycle=0.15, prefill=6,  decode=117),
                "tool_m":  AgentConfig(size=14, cycle=1.21, prefill=12, decode=182),
                "tool_l":  AgentConfig(size=32, cycle=1.82, prefill=16, decode=241),
            },

            "LongBench": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=2,  decode=96),
                "critic":  AgentConfig(size=32, cycle=0.48, prefill=6,  decode=121),
                "tool_s":  AgentConfig(size=8,  cycle=0.21, prefill=4,  decode=88),
                "tool_m":  AgentConfig(size=14, cycle=0.82, prefill=8,  decode=158),
                "tool_l":  AgentConfig(size=32, cycle=1.03, prefill=12, decode=207),
            },

            "BBH": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=2,  decode=92),
                "critic":  AgentConfig(size=32, cycle=0.7, prefill=4,  decode=130),
                "tool_s":  AgentConfig(size=8,  cycle=0.11, prefill=2, decode=100),
                "tool_m":  AgentConfig(size=14, cycle=0.88, prefill=4, decode=170),
                "tool_l":  AgentConfig(size=32, cycle=1.21, prefill=6, decode=210),
            },

            "ShareGPT": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=2,  decode=98),
                "critic":  AgentConfig(size=32, cycle=0.21, prefill=4,  decode=118),
                "tool_s":  AgentConfig(size=8,  cycle=0.91, prefill=2,  decode=121),
                "tool_m":  AgentConfig(size=14, cycle=0.62, prefill=4,  decode=139),
                "tool_l":  AgentConfig(size=32, cycle=0.12, prefill=8,  decode=182),
            },

            "SWE-bench": {
                "planner": AgentConfig(size=32, cycle=1.2, prefill=8,  decode=81),
                "critic":  AgentConfig(size=32, cycle=1,   prefill=16, decode=158),
                "tool_s":  AgentConfig(size=8,  cycle=0.25, prefill=8, decode=90),
                "tool_m":  AgentConfig(size=14, cycle=1.2, prefill=12, decode=120),
                "tool_l":  AgentConfig(size=32, cycle=1.75, prefill=24, decode=177),
            },

            "HumanEval": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=2,  decode=48),
                "critic":  AgentConfig(size=32, cycle=0.51, prefill=4,  decode=81),
                "tool_s":  AgentConfig(size=8,  cycle=0.36, prefill=2,  decode=97),
                "tool_m":  AgentConfig(size=14, cycle=0.92, prefill=4,  decode=117),
                "tool_l":  AgentConfig(size=32, cycle=0.52, prefill=8,  decode=153),
            },

            "DS-1000": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=2,  decode=79),
                "critic":  AgentConfig(size=32, cycle=0.88, prefill=8,  decode=96),
                "tool_s":  AgentConfig(size=8,  cycle=0.39, prefill=4,  decode=90),
                "tool_m":  AgentConfig(size=14, cycle=1.03, prefill=8,  decode=122),
                "tool_l":  AgentConfig(size=32, cycle=1.13, prefill=12, decode=187),
            },

            "MBPP": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=1,   decode=48),
                "critic":  AgentConfig(size=32, cycle=0.3, prefill=2,   decode=64),
                "tool_s":  AgentConfig(size=8,  cycle=0.76, prefill=0.5, decode=48),
                "tool_m":  AgentConfig(size=14, cycle=0.53, prefill=2,   decode=90),
                "tool_l":  AgentConfig(size=32, cycle=0.21, prefill=4,   decode=128),
            },

            "DocVQA": {
                "planner": AgentConfig(size=32, cycle=1.21, prefill=2,  decode=92),
                "critic":  AgentConfig(size=32, cycle=0.99, prefill=12, decode=170),
                "tool_s":  AgentConfig(size=8,  cycle=0.23, prefill=4,  decode=120),
                "tool_m":  AgentConfig(size=14, cycle=0.61, prefill=8,  decode=168),
                "tool_l":  AgentConfig(size=32, cycle=1.17, prefill=16, decode=220),
            },

            "MMBench": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=1,  decode=63),
                "critic":  AgentConfig(size=32, cycle=0.2, prefill=2,  decode=81),
                "tool_s":  AgentConfig(size=8,  cycle=0.85, prefill=2, decode=71),
                "tool_m":  AgentConfig(size=14, cycle=0.66, prefill=3, decode=87),
                "tool_l":  AgentConfig(size=32, cycle=0.41, prefill=6, decode=112),
            },

            "MMMU": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=1,  decode=80),
                "critic":  AgentConfig(size=32, cycle=0.52, prefill=4,  decode=127),
                "tool_s":  AgentConfig(size=8,  cycle=0.16, prefill=2,  decode=77),
                "tool_m":  AgentConfig(size=14, cycle=0.61, prefill=4,  decode=143),
                "tool_l":  AgentConfig(size=32, cycle=0.74, prefill=6,  decode=192),
            },

            "VQAv2": {
                "planner": AgentConfig(size=32, cycle=1,   prefill=2,  decode=40),
                "critic":  AgentConfig(size=32, cycle=0.15, prefill=2,  decode=49),
                "tool_s":  AgentConfig(size=8,  cycle=0.77, prefill=2,  decode=42),
                "tool_m":  AgentConfig(size=14, cycle=0.48, prefill=3,  decode=63),
                "tool_l":  AgentConfig(size=32, cycle=0.25, prefill=6,  decode=101),
            },
        }

    def __getitem__(self, dataset: str) -> Dict[str, AgentConfig]:
        if dataset not in self.agent_config:
            valid = ", ".join(self.agent_config.keys())
            raise KeyError(
                f"Unknown dataset '{dataset}'. "
                f"Valid datasets are: {valid}"
            )
        return self.agent_config[dataset]

default_agent_config = AgentConfigStore()

MAX_OFF_PCLE_BW_UTIL = 0.85

def load_csv_or_error(filename: str):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File {filename} not found. Please first run real_vllmtest.py or place the corresponding file in this folder.")
    return pd.read_csv(path)

model_8B = load_csv_or_error("./src/model_size8B.csv")
model_14B = load_csv_or_error("./src/model_size14B.csv")
model_32B = load_csv_or_error("./src/model_size32B.csv")

def get_prefill_time(dataset: str,device: str, config = default_agent_config):
    if device in ("H100", "AgentX"):
        length_map = default_agent_config.length_map
        config = config[dataset]
        prefill_latency = model_32B.loc[length_map[config["planner"].prefill], "prefill"] * config["planner"].cycle + \
                          model_32B.loc[length_map[config["critic"].prefill], "prefill"] * config["critic"].cycle + \
                          model_8B.loc[length_map[config["tool_s"].prefill], "prefill"] * config["tool_s"].cycle + \
                          model_14B.loc[length_map[config["tool_m"].prefill], "prefill"] * config["tool_m"].cycle + \
                          model_32B.loc[length_map[config["tool_l"].prefill], "prefill"] * config["tool_l"].cycle
        return prefill_latency
    else:
        raise ValueError(f"Unknown device type: {device}")
    
def get_pcle_time(dataset: str,device: str,config = default_agent_config):
    if device == "H100":
        pcle = default_agent_config.pcle * MAX_OFF_PCLE_BW_UTIL
        config = config[dataset]
        pcle_latency = 2 * ( config["planner"].size * config["planner"].cycle + \
                           config["critic"].size * config["critic"].cycle + \
                           config["tool_s"].size * config["tool_s"].cycle + \
                           config["tool_m"].size * config["tool_m"].cycle + \
                           config["tool_l"].size * config["tool_l"].cycle ) / pcle
        return pcle_latency
    else:
        raise ValueError(f"Unknown device type: {device}")

def get_decode_time(dataset: str,device: str,config = default_agent_config):
    if device == "H100":
        length_map = default_agent_config.length_map
        config = config[dataset]
        decode_latency = (config["planner"].decode * config["planner"].cycle * model_32B.loc[length_map[config["planner"].prefill], "decode"] + \
                          config["critic"].decode * config["critic"].cycle * model_32B.loc[length_map[config["critic"].prefill], "decode"] + \
                          config["tool_s"].decode * config["tool_s"].cycle * model_8B.loc[length_map[config["tool_s"].prefill], "decode"] + \
                          config["tool_m"].decode * config["tool_m"].cycle * model_14B.loc[length_map[config["tool_m"].prefill], "decode"] + \
                          config["tool_l"].decode * config["tool_l"].cycle * model_32B.loc[length_map[config["tool_l"].prefill], "decode"]) / 1000
        return decode_latency
    else:
        raise ValueError(f"Unknown device type: {device}")

def get_AgentX_time(dataset: str,device: str,batch_size: int, maxlen: int, dbyte: int, run_lpddrpim, config = default_agent_config):
    if device == "AgentX":
        config = config[dataset]
        tck_ns = 0.3125  # 6400MT/s -> 0.3125ns per tick
        
        modelsize = str(config["planner"].size) + "B"
        context_len = config["planner"].prefill * 1024
        layer = model_config[modelsize]["layer"]
        decode_len = config["planner"].decode
        cycle = config["planner"].cycle
        planner_decode_cycles = run_lpddrpim(modelsize, context_len, batch_size, maxlen, dbyte)
        planner_decode_time = (planner_decode_cycles * tck_ns * layer * decode_len * cycle * 2) / 1e9
        
        modelsize = str(config["critic"].size) + "B"
        context_len = config["critic"].prefill * 1024
        layer = model_config[modelsize]["layer"]
        decode_len = config["critic"].decode
        cycle = config["critic"].cycle
        critic_decode_cycles = run_lpddrpim(modelsize, context_len, batch_size, maxlen, dbyte)
        critic_decode_time = (critic_decode_cycles * tck_ns * layer * decode_len * cycle * 2) / 1e9
        
        modelsize = str(config["tool_s"].size) + "B"
        context_len = config["tool_s"].prefill * 1024
        layer = model_config[modelsize]["layer"]
        decode_len = config["tool_s"].decode
        cycle = config["tool_s"].cycle
        tool_s_decode_cycles = run_lpddrpim(modelsize, context_len, batch_size, maxlen, dbyte)
        tool_s_decode_time = (tool_s_decode_cycles * tck_ns * layer * decode_len * cycle * 2) / 1e9
        
        modelsize = str(config["tool_m"].size) + "B"
        context_len = config["tool_m"].prefill * 1024
        layer = model_config[modelsize]["layer"]
        decode_len = config["tool_m"].decode
        cycle = config["tool_m"].cycle
        tool_m_decode_cycles = run_lpddrpim(modelsize, context_len, batch_size, maxlen, dbyte)
        tool_m_decode_time = (tool_m_decode_cycles * tck_ns * layer * decode_len * cycle * 2) / 1e9
        
        modelsize = str(config["tool_l"].size) + "B"
        context_len = config["tool_l"].prefill * 1024
        layer = model_config[modelsize]["layer"]
        decode_len = config["tool_l"].decode
        cycle = config["tool_l"].cycle
        tool_l_decode_cycles = run_lpddrpim(modelsize, context_len, batch_size, maxlen, dbyte)
        tool_l_decode_time = (tool_l_decode_cycles * tck_ns * layer * decode_len * cycle * 2) / 1e9
        
        total_decode_time = planner_decode_time + critic_decode_time + tool_s_decode_time + tool_m_decode_time + tool_l_decode_time
        return total_decode_time
    else:
        raise ValueError(f"Unknown device type: {device}")