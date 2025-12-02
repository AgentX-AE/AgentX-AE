import os
import time
import random
import string
import argparse
import csv
from typing import Dict, List, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def measure_prefill_and_decode_for_contexts(
    model_name: str,
    model_path: str,
    ctx_list_tokens: List[int],
    device: str = "0",
    rounds: int = 5,
    max_tokens_decode: int = 128,
) -> Dict[int, Tuple[float, float]]:
    """
    Measure *real* prefill (TTFT) and per-token decode latency for a list of
    context lengths (in tokens).

    For each context length L, we estimate:
      - TTFT_est_avg(L): average prefill time in seconds
      - TPOT_est_avg(L): average decode latency in ms/token

    Returns:
        {
            L_tokens_1: (ttft_s_1, decode_ms_token_1),
            L_tokens_2: (ttft_s_2, decode_ms_token_2),
            ...
        }
    """

    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # Safety margin: real prompt length = (target - SAFETY)
    SAFETY = 32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    # Create vLLM engine
    # IMPORTANT: disable prefix caching to get *real* prefill time
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=max(ctx_list_tokens) + 512,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
    )

    # Character set used for random prompt generation
    CHARS = string.ascii_letters + string.digits + " ,.;:?!()[]{}+-*/=_"

    def random_prompt_by_exact_tokens(n_tokens: int) -> str:
        """
        Generate a random prompt which tokenizes to approximately n_tokens.
        We intentionally keep a SAFETY margin to avoid going over max length.
        """
        tokens: List[int] = []
        while len(tokens) < n_tokens:
            chunk = "".join(random.choices(CHARS, k=32))
            # encode without special tokens to make counting easier
            chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)
            tokens.extend(chunk_ids)

        target_len = max(1, n_tokens - SAFETY)
        tokens_trimmed = tokens[:target_len]
        prompt_text = tokenizer.decode(tokens_trimmed)
        return prompt_text

    # Global warm-up: small prompt just to initialize kernels, CUDA graphs, etc.
    llm.generate(["warmup"], SamplingParams(max_tokens=1, temperature=0.0))

    print(f"\n===== [{model_name}] REAL PREFILL & DECODE LATENCY TEST =====")
    print(f"Model path: {model_path}")
    print("====================================================\n")

    results: Dict[int, Tuple[float, float]] = {}

    for target_ctx in ctx_list_tokens:
        prompt = random_prompt_by_exact_tokens(target_ctx)

        # Per-context warm-up (NOT recorded)
        params_one = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )
        params_N = SamplingParams(
            max_tokens=max_tokens_decode,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )

        # Warm-up to absorb one-time overhead for this context length
        _ = llm.generate([prompt], params_one)
        _ = llm.generate([prompt], params_N)

        sum_ttft_est = 0.0       # sum of estimated prefill (seconds)
        sum_tpot = 0.0           # sum of per-token decode (seconds)
        sum_gen_tokens = 0.0     # sum of N across rounds
        real_ctx_last = 0
        valid_rounds = 0

        for r in range(rounds):
            # 1) Run with max_tokens = 1  --> prefill + 1 token
            t0 = time.perf_counter()
            out_one = llm.generate([prompt], params_one)
            t1 = time.perf_counter()
            t_one = t1 - t0

            # 2) Run with max_tokens = max_tokens_decode --> prefill + N tokens
            t0 = time.perf_counter()
            out_N = llm.generate([prompt], params_N)
            t1 = time.perf_counter()
            t_many = t1 - t0

            gen_tokens = len(out_N[0].outputs[0].token_ids)
            if gen_tokens <= 1:
                print(
                    f"[ctx={target_ctx:5d}] Round {r+1}: "
                    f"generated only {gen_tokens} tokens, skip this round."
                )
                continue

            N = gen_tokens
            real_ctx = len(out_N[0].prompt_token_ids)
            real_ctx_last = real_ctx

            # Two-point approximation:
            #   t_one  ≈ P_eff + d
            #   t_many ≈ P_eff + N * d
            # => d     = (t_many - t_one) / (N - 1)
            #    P_eff = t_one - d
            d = (t_many - t_one) / (N - 1)     # per-token decode time (seconds)
            P_eff = t_one - d                  # estimated prefill time (seconds)

            sum_ttft_est += P_eff
            sum_tpot += d
            sum_gen_tokens += N
            valid_rounds += 1

            print(
                f"[ctx={target_ctx:5d}] Round {r+1}: "
                f"t1={t_one:.4f}s, tN={t_many:.4f}s, "
                f"TTFT_est={P_eff:.4f}s, "
                f"TPOT_est≈{d*1000:.2f} ms/token, N={N}"
            )

        if valid_rounds == 0:
            print(f"[ctx={target_ctx:5d}] all rounds skipped (N<=1).")
            print("-" * 80)
            continue

        ttft_avg = sum_ttft_est / valid_rounds      # seconds
        tpot_avg = sum_tpot / valid_rounds          # seconds/token

        print(f"\nContext target = {target_ctx:5d}, real ≈ {real_ctx_last}")
        print(f"  TTFT_est (avg): {ttft_avg:.4f} s")
        print(f"  TPOT_est (avg): {tpot_avg*1000:.2f} ms/token")
        print("-" * 80)

        # Store: TTFT in seconds, decode in ms/token
        results[target_ctx] = (ttft_avg, tpot_avg * 1000.0)

    # Summary table
    print("\n========== SUMMARY (REAL PREFILL & DECODE PER CONTEXT) ==========\n")
    print(f"{'CtxTok':>8} | {'TTFT(s)':>10} | {'TPOT(ms)':>10}")
    print("-" * 40)
    for ctx in sorted(results.keys()):
        ttft_s, decode_ms = results[ctx]
        print(f"{ctx:8d} | {ttft_s:10.4f} | {decode_ms:10.2f}")
    print("-" * 40)
    print()

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure prefill & decode latency (0.5k-24k) and write CSV."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Logical model name (for logging only).",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Local model path or HF repo id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id (as string), e.g., '0' or '1'.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="Model size string for output file name, e.g., 32B, 14B, 8B. "
             "If not set, will use `name`.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 0.5k–24k contexts → length_k = 0.5,1,2,3,4,6,8,12,16,24
    lengths_k = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24]
    ctx_tokens = [int(l * 1024) for l in lengths_k]

    # Measure per-context TTFT + decode(ms/token)
    stats = measure_prefill_and_decode_for_contexts(
        model_name=args.name,
        model_path=args.path,
        ctx_list_tokens=ctx_tokens,
        device=args.device,
        rounds=5,
        max_tokens_decode=128,
    )

    # CSV file name: mode_sizeXX.csv  (XX = model_size or name)
    model_size_str = args.model_size if args.model_size is not None else args.name
    out_filename = f"model_size{model_size_str}.csv"

    # Write CSV: length,prefill,decode  (length in k tokens)
    import csv
    with open(out_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["length", "prefill", "decode"])
        for length_k, ctx_tok in zip(lengths_k, ctx_tokens):
            if ctx_tok not in stats:
                continue
            ttft_s, decode_ms = stats[ctx_tok]
            writer.writerow([length_k, ttft_s, decode_ms])

    print(f"\nCSV written to: {out_filename}")
    print("Format: length (k), prefill (s), decode (ms/token)")

