import math, time, warnings, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
SEQ_LENS = [10, 100, 1000, 10_000]       # input lengths to test
D_MODEL  = 64                             # hidden size
N_HEADS  = 4                              # number of heads (assume d_k=d_v=D_MODEL//N_HEADS)
BATCH    = 1
TARGET_TOTAL_TIME_S = 0.8                 # try to spend at least ~0.8s per (len, device) for small SE
WARMUP_ROUNDS = 3
PLOT_DPI = 160
TORCH_SEED = 1234

# ---------------------------
# Self-Attention (naive)
# ---------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        H, d_k = self.n_heads, self.d_k

        Q = self.q_proj(x).reshape(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.k_proj(x).reshape(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        V = self.v_proj(x).reshape(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]

        # Scaled dot-product attention (naive, full n^2)
        # attn_logits: [B, H, N, N]
        attn_logits = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
        attn_probs = torch.softmax(attn_logits, dim=-1)           # [B, H, N, N]
        context = attn_probs @ V                                  # [B, H, N, d_k]

        context = context.transpose(1, 2).reshape(B, N, D)        # [B, N, D]
        out = self.o_proj(context)                                # [B, N, D]
        return out

# ---------------------------
# FLOPs (analytical)
# ---------------------------
def attention_flops(n: int, d_model: int, n_heads: int) -> float:
    """
    Analytical FLOPs for naive forward pass (multiply+add counted as 2 FLOPs).
    Assumes d_k = d_v = d_model // n_heads.
    Includes: Q/K/V projections, attention score matmul, softmax (approx), attn*V, output projection.
    """
    d_k = d_model // n_heads
    # Linear projections: 4 matrices (Q,K,V,Out), each ~ 2*n*d_model*d_model
    flops_proj = 4 * (2 * n * d_model * d_model)

    # Scores QK^T: per head 2*n*n*d_k, across heads:
    flops_scores = 2.0 * n * n * d_k * n_heads

    # Softmax approx cost: per head, per row length n: ~5n ops => ~5*n*n per head
    flops_softmax = 5.0 * n * n * n_heads

    # Attention*V: per head 2*n*n*d_k
    flops_attn_v = 2.0 * n * n * d_k * n_heads

    total = flops_proj + flops_scores + flops_softmax + flops_attn_v
    return float(total)

# ---------------------------
# Memory (estimated for CPU; measured on GPU)
# ---------------------------
def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8

def attention_activation_bytes(n: int, d_model: int, n_heads: int, dtype: torch.dtype) -> int:
    """
    Estimate peak activation memory (bytes) for the forward pass (not counting params) for B=1:
      Q, K, V: 3 * [N, D]
      attn_logits: [H, N, N]
      attn_probs:  [H, N, N]
      context:     [N, D]
    """
    d_k = d_model // n_heads
    bytes_per = dtype_nbytes(dtype)
    qkv = 3 * (n * d_model)
    logits = n_heads * (n * n)
    probs = n_heads * (n * n)
    context = n * d_model
    total_elems = qkv + logits + probs + context
    return int(total_elems * bytes_per)

# ---------------------------
# Benchmark helpers
# ---------------------------
@dataclass
class Stat:
    mean: float
    se: float

def standard_error(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.std(xs, ddof=1) / math.sqrt(len(xs)))

def calibrate_repeats(fn, target_s: float, max_repeats: int = 200) -> int:
    # quick timing to choose repeats that reach ~target_s
    t0 = time.perf_counter()
    count = 0
    while time.perf_counter() - t0 < max(0.05, target_s / 5) and count < 5:
        fn()
        count += 1
    elapsed = time.perf_counter() - t0
    per = elapsed / max(1, count)
    # choose repeats to get near target time
    reps = int(max(3, min(max_repeats, math.ceil(target_s / max(1e-5, per)))))
    return reps

def run_one_length(model: nn.Module, n: int, device: torch.device, dtype: torch.dtype) -> Tuple[float, int]:
    """
    Run a single forward, return (elapsed_seconds, peak_memory_bytes).
    On GPU: memory is measured via CUDA peak stats.
    On CPU: memory is estimated analytically.
    """
    model.eval()
    with torch.inference_mode():
        x = torch.randn(BATCH, n, D_MODEL, device=device, dtype=dtype)
        # memory tracking
        peak_mem = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            y = model(x)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)  # milliseconds
            peak_mem = torch.cuda.max_memory_allocated(device)
            elapsed_s = ms / 1000.0
        else:
            # CPU timing
            t0 = time.perf_counter()
            y = model(x)
            elapsed_s = time.perf_counter() - t0
            # CPU memory: estimate activations (params excluded)
            peak_mem = attention_activation_bytes(n, D_MODEL, N_HEADS, dtype)
        del y, x
        return elapsed_s, int(peak_mem)

def benchmark(device: torch.device, dtype: torch.dtype) -> Dict[str, Dict[int, Stat]]:
    torch.manual_seed(TORCH_SEED)
    model = MultiHeadSelfAttention(D_MODEL, N_HEADS).to(device=device, dtype=dtype)

    # warmup (important for GPU clocks, caching allocator, etc.)
    for _ in range(WARMUP_ROUNDS):
        _ = run_one_length(model, 128, device, dtype)

    time_means, time_ses = {}, {}
    mem_means, mem_ses   = {}, {}
    flops_vals           = {}

    for n in SEQ_LENS:
        # early OOM guard for GPU: try a dry run
        try:
            _ = run_one_length(model, min(n, 64), device, dtype)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                warnings.warn(f"OOM during warmup at n={n} on {device}; skipping.")
                continue

        # choose repeats to shrink SE
        def one():
            run_one_length(model, n, device, dtype)
        repeats = calibrate_repeats(one, TARGET_TOTAL_TIME_S)

        times, mems = [], []
        for _ in range(repeats):
            try:
                t_s, peak_b = run_one_length(model, n, device, dtype)
            except RuntimeError as e:
                if device.type == "cuda" and "CUDA out of memory" in str(e):
                    warnings.warn(f"OOM at n={n} on {device}; skipping remaining reps.")
                    break
                else:
                    raise
            times.append(t_s)
            mems.append(peak_b)

        if len(times) == 0:
            continue

        time_means[n] = Stat(mean=float(np.mean(times)), se=standard_error(times))
        mem_means[n]  = Stat(mean=float(np.mean(mems)),  se=standard_error(mems))
        flops_vals[n] = attention_flops(n, D_MODEL, N_HEADS)

    return {
        "time_mean": time_means,
        "mem_mean": mem_means,
        "flops": {n: Stat(mean=v, se=0.0) for n, v in flops_vals.items()}  # deterministic
    }

# ---------------------------
# Plotting
# ---------------------------
def plot_metric(xs: List[int],
                cpu_stats: Dict[int, Stat],
                gpu_stats: Dict[int, Stat],
                title: str, ylabel: str, yscale: str, transform=lambda v: v,
                filename: str = "plot.png"):
    plt.figure(figsize=(6.0, 4.0), dpi=PLOT_DPI)
    # CPU
    x_cpu = [n for n in xs if n in cpu_stats]
    y_cpu = [transform(cpu_stats[n].mean) for n in x_cpu]
    e_cpu = [transform(cpu_stats[n].se) for n in x_cpu]
    plt.errorbar(x_cpu, y_cpu, yerr=e_cpu, fmt='-o', label='CPU', capsize=3)

    # GPU
    x_gpu = [n for n in xs if n in gpu_stats]
    y_gpu = [transform(gpu_stats[n].mean) for n in x_gpu]
    e_gpu = [transform(gpu_stats[n].se) for n in x_gpu]
    plt.errorbar(x_gpu, y_gpu, yerr=e_gpu, fmt='-s', label='GPU', capsize=3)

    plt.xscale('log')
    if yscale:
        plt.yscale(yscale)
    plt.xticks(xs, [str(x) for x in xs])
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")

def main():
    print(f"PyTorch: {torch.__version__}")
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    cpu = torch.device('cpu')
    # Use fp32 on CPU; fp16/bf16 on GPU for realism & to fit larger N
    cpu_dtype = torch.float32

    # CPU benchmark
    print("\n=== CPU benchmark ===")
    cpu_res = benchmark(cpu, cpu_dtype)

    # GPU benchmark (optional)
    if has_cuda:
        gpu = torch.device('cuda')
        # Prefer float16 when available
        gpu_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        torch.cuda.empty_cache()
        print("\n=== GPU benchmark ===")
        gpu_res = benchmark(gpu, gpu_dtype)
    else:
        gpu_res = {"time_mean": {}, "mem_mean": {}, "flops": {}}

    # Collate and plot
    xs = SEQ_LENS

    # Time (ms)
    cpu_time = cpu_res["time_mean"]
    gpu_time = gpu_res["time_mean"]
    plot_metric(xs, cpu_time, gpu_time,
                title="Self-Attention Wall Time vs Length",
                ylabel="Time (ms)",
                yscale="log",
                transform=lambda s: s * 1e3,
                filename="time_ms_vs_len.png")

    # Memory (GiB)
    cpu_mem = cpu_res["mem_mean"]
    gpu_mem = gpu_res["mem_mean"]
    plot_metric(xs, cpu_mem, gpu_mem,
                title="Self-Attention Peak Memory vs Length",
                ylabel="Peak activation memory (GiB)",
                yscale="log",
                transform=lambda b: b / (1024**3),
                filename="memory_gib_vs_len.png")

    # FLOPs (GFLOPs) – deterministic; SE=0
    cpu_flops = {n: Stat(mean=cpu_res["flops"].get(n, Stat(0,0)).mean, se=0.0) for n in xs}
    gpu_flops = {n: Stat(mean=gpu_res["flops"].get(n, Stat(0,0)).mean, se=0.0) for n in xs}
    plot_metric(xs, cpu_flops, gpu_flops,
                title="Self-Attention Theoretical FLOPs vs Length",
                ylabel="FLOPs (GFLOPs)",
                yscale="log",
                transform=lambda f: f / 1e9,
                filename="flops_gflops_vs_len.png")

    # Brief printed summary
    def table(stats: Dict[int, Stat], label: str, unit: str, tf=lambda v: v):
        rows = []
        for n in xs:
            if n in stats:
                s = stats[n]
                rows.append((n, tf(s.mean), tf(s.se)))
        if rows:
            print(f"\n{label} (mean ± SE) [{unit}]")
            for n, m, e in rows:
                print(f"  N={n:<5} {m:.3g} ± {e:.2g}")

    table(cpu_time, "CPU time", "ms", lambda s: s*1e3)
    table(gpu_time, "GPU time", "ms", lambda s: s*1e3)
    table(cpu_mem, "CPU peak activations (est.)", "GiB", lambda b: b/(1024**3))
    table(gpu_mem, "GPU peak activations (meas.)", "GiB", lambda b: b/(1024**3))

if __name__ == "__main__":
    # ensure deterministic-ish behavior
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
