# perplexity_and_sampling.py
# Requirements: transformers>=4.40, torch, numpy, matplotlib (installed), GPU optional but used if available.

import os, random, math, argparse
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# -------------------------
# Utilities
# -------------------------
def set_device_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, dtype

def distinct_n(text: str, n: int = 1) -> float:
    toks = text.strip().split()
    if len(toks) < n:
        return 0.0
    total = len(toks) - n + 1
    ngrams = {" ".join(toks[i:i+n]) for i in range(total)}
    return len(ngrams) / max(1, total)

def shuffle_words(text: str, seed: int = 1337) -> str:
    toks = text.split()
    rng = random.Random(seed)
    rng.shuffle(toks)
    return " ".join(toks)

def compute_perplexity(text: str, model, tokenizer, device) -> float:
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        # GPT-2 family has no pad token by default; align to eos for safety
        labels = input_ids.clone()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        # out.loss is mean NLL per token (in nats if model uses log-softmax base e); exp => perplexity
        ppl = float(torch.exp(out.loss).item())
    return ppl

def generate_text(prompt: str, model, tokenizer, device, max_new_tokens=500,
                  do_sample=False, temperature=1.0) -> str:
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

# -------------------------
# Main experiment
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)
    device, dtype = set_device_dtype()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # (a) Perplexity Analysis
    paragraph = (
        "Language models learn statistical patterns in text and use them to predict what comes next. "
        "They are powerful at producing fluent language but can sometimes make mistakes. "
        "Evaluating them requires both automatic metrics and careful human judgment. "
        "Perplexity is a common metric that reflects how well a model predicts a sequence."
    )
    shuffled = shuffle_words(paragraph, seed=args.seed)

    ppl_orig = compute_perplexity(paragraph, model, tokenizer, device)
    ppl_shuf = compute_perplexity(shuffled, model, tokenizer, device)

    print("\n=== (a) Perplexity Analysis ===")
    print(f"Original paragraph:\n{paragraph}\n")
    print(f"Shuffled paragraph:\n{shuffled}\n")
    print(f"Perplexity (original): {ppl_orig:.3f}")
    print(f"Perplexity (shuffled): {ppl_shuf:.3f}")
    print("Comment: Shuffling breaks syntax/semantics -> typically higher perplexity on shuffled text.\n")

    # (b) Sampling Comparison
    os.makedirs("outputs", exist_ok=True)
    prompt = "Once upon a time"
    temps = [0, 0.3, 0.6, 0.9, 1.2, 1.5]

    print("=== (b) Sampling Comparison (max_new_tokens = {}) ===".format(args.max_new_tokens))
    all_outputs: List[Tuple[str, str, float, float, int]] = []

    # Greedy decoding (equivalent to temperature 0 but clearer to show both)
    greedy_out = generate_text(prompt, model, tokenizer, device,
                               max_new_tokens=args.max_new_tokens, do_sample=False)
    d1 = distinct_n(greedy_out, 1)
    d2 = distinct_n(greedy_out, 2)
    all_outputs.append(("greedy", greedy_out, d1, d2, len(greedy_out.split())))
    print("\n--- Greedy Decoding ---")
    print(greedy_out)
    print(f"\n[greedy] distinct-1={d1:.3f} distinct-2={d2:.3f} length_tokens≈{len(greedy_out.split())}")

    # Temperature sampling
    for T in temps:
        if T <= 0:
            label = "temp=0 (greedy-equivalent)"
            out = generate_text(prompt, model, tokenizer, device,
                                max_new_tokens=args.max_new_tokens, do_sample=False)
        else:
            label = f"temp={T}"
            out = generate_text(prompt, model, tokenizer, device,
                                max_new_tokens=args.max_new_tokens, do_sample=True, temperature=T)
        d1 = distinct_n(out, 1)
        d2 = distinct_n(out, 2)
        all_outputs.append((label, out, d1, d2, len(out.split())))
        print(f"\n--- {label} ---")
        print(out)
        print(f"\n[{label}] distinct-1={d1:.3f} distinct-2={d2:.3f} length_tokens≈{len(out.split())}")

    # Save to disk
    with open(os.path.join("outputs", "sampling_outputs.txt"), "w", encoding="utf-8") as f:
        f.write("Prompt: " + prompt + "\n\n")
        for label, out, d1, d2, L in all_outputs:
            f.write(f"==== {label} ====\n")
            f.write(out + "\n")
            f.write(f"\n[{label}] distinct-1={d1:.3f} distinct-2={d2:.3f} length_tokens≈{L}\n\n")

    # Brief comparison
    print("\nSummary:")
    for label, _, d1, d2, L in all_outputs:
        print(f"{label:>16} | distinct-1={d1:.3f} distinct-2={d2:.3f} len={L}")
    print("\nBrief note: Lower temperatures (incl. greedy) are more deterministic and repetitive; "
          "higher temperatures increase diversity (higher distinct-n) but can reduce coherence.")

if __name__ == "__main__":
    main()
