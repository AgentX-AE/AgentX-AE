model_config = {
    "8B": {
        "layer":    36,
        "d_model":  4096,
        "d_ff":     12288,
        "n_heads":  32,
        "n_kv":     8,
        "d_head":   128,
    },
    "14B": {
        "layer":    40,
        "d_model":  5120,
        "d_ff":     17408,
        "n_heads":  40,
        "n_kv":     8,
        "d_head":   128,
    },
    "32B": {
        "layer":    64,
        "d_model":  5120,
        "d_ff":     25600,
        "n_heads":  64,
        "n_kv":     8,
        "d_head":   128,
    },
    "70B": {
        "layer":    80,
        "d_model":  8192,
        "d_ff":     28672,
        "n_heads":  64,
        "n_kv":     8,
        "d_head":   128,
    },
    
}

def get_decode_shapes(model_name: str, batch_size: int, context_len: int):
    cfg = model_config[model_name.upper()]
    d_model = cfg["d_model"]
    d_ff    = cfg["d_ff"]
    n_heads = cfg["n_heads"]
    n_kv    = cfg["n_kv"]
    d_head  = cfg["d_head"]

    shapes = {
        "meta": {
            "layer":     cfg["layer"],
            "d_model":   d_model,
            "d_ff":      d_ff,
            "n_heads":   n_heads,
            "n_kv":      n_kv,
            "d_head":    d_head,
            "batch":     batch_size,
            "context":   context_len,
        },

        # ========= Stage 1: Q / K / V projection =========
        "q_proj": {
            "input":  [batch_size, d_model],
            "weight": [d_model, n_heads * d_head],     
            "output": [batch_size, n_heads * d_head],  
        },
        "k_proj": {
            "input":  [batch_size, d_model],
            "weight": [d_model, n_kv * d_head],        
            "output": [batch_size, n_kv * d_head],
        },
        "v_proj": {
            "input":  [batch_size, d_model],
            "weight": [d_model, n_kv * d_head],
            "output": [batch_size, n_kv * d_head],
        },

        # ========= Stage 2: Attention=========
        "attn_qk": {
            "weight":  [batch_size * d_head, context_len],
        },
        "attn_av": {
            "matmul_v":     [context_len, batch_size * d_head],
        },
        # O projection
        "o_proj": {
            "input":  [batch_size, n_heads * d_head],  
            "weight": [n_heads * d_head, d_model], 
            "output": [batch_size, d_model],
        },

        # ========= Stage 3: FFN (SwiGLU) =========
        "gate_proj": {
            "input":  [batch_size, d_model],
            "weight": [d_model, d_ff],                 
            "output": [batch_size, d_ff],
        },
        "up_proj": {
            "input":  [batch_size, d_model],
            "weight": [d_model, d_ff],
            "output": [batch_size, d_ff],
        },
        "down_proj": {
            "input":  [batch_size, d_ff],
            "weight": [d_ff, d_model],
            "output": [batch_size, d_model],
        },
    }

    return shapes

# print(default_agent_config.agent_config["SWE-bench"]["planner"].decode)
