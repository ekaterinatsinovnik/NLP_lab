import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

CHOICES = ["A", "B", "C", "D"]
WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth"}

def load_tokenizer_safe(path_or_id: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(path_or_id, trust_remote_code=trust_remote_code, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(path_or_id, trust_remote_code=trust_remote_code)

def list_subjects(mmlu_root: Path) -> List[str]:
    test_dir = mmlu_root / "data" / "test"
    return sorted([p.name.replace("_test.csv", "") for p in test_dir.glob("*_test.csv")])

def read_csv_rows(path: Path) -> List[Tuple[str, str, str, str, str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 6:
                continue
            rows.append((r[0], r[1], r[2], r[3], r[4], r[5]))
    return rows

def preload_mmlu_csvs(mmlu_root: Path, subjects: List[str], max_workers: int = 8):
    dev_dir = mmlu_root / "data" / "dev"
    test_dir = mmlu_root / "data" / "test"
    dev_map, test_map = {}, {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for subj in subjects:
            futures[("dev", subj)] = ex.submit(read_csv_rows, dev_dir / f"{subj}_dev.csv")
            futures[("test", subj)] = ex.submit(read_csv_rows, test_dir / f"{subj}_test.csv")
        for (kind, subj), fut in futures.items():
            if kind == "dev":
                dev_map[subj] = fut.result()
            else:
                test_map[subj] = fut.result()
    return dev_map, test_map

def subject_to_title(subj: str) -> str:
    return subj.replace("_", " ")

def format_example(question: str, a: str, b: str, c: str, d: str, answer: Optional[str]) -> str:
    s = [
        f"Question: {question}",
        f"A. {a}",
        f"B. {b}",
        f"C. {c}",
        f"D. {d}",
        "Answer:" if answer is None else f"Answer: {answer}",
    ]
    return "\n".join(s)

def build_fewshot_prompt(dev_rows, test_row, subj: str, k: int) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject_to_title(subj)}.\n\n"
    parts = [header]
    for i in range(min(k, len(dev_rows))):
        q, a, b, c, d, ans = dev_rows[i]
        parts.append(format_example(q, a, b, c, d, ans.strip()))
        parts.append("")
    tq, ta, tb, tc, td, _ = test_row
    parts.append(format_example(tq, ta, tb, tc, td, answer=None))
    return "\n".join(parts).rstrip() + " "

def weight_files_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in WEIGHT_EXTS:
            total += p.stat().st_size
    return total

def _dequant_fp8_to_dtype(q_fp8: torch.Tensor, scale: torch.Tensor, scheme: str, out_dtype: torch.dtype) -> torch.Tensor:
    q = q_fp8.to(torch.float32)
    if scheme == "per_row" and q.ndim == 2:
        w = q * scale.to(torch.float32).unsqueeze(1)
    elif scheme == "per_tensor":
        w = q * scale.to(torch.float32)
    else:
        w = q
    return w.to(out_dtype)

def load_fp8_scaled_dequant_once(fp8_dir: Path, trust_remote_code: bool, device: str, out_dtype: torch.dtype):
    cfg = AutoConfig.from_pretrained(fp8_dir, trust_remote_code=trust_remote_code)
    tok = load_tokenizer_safe(str(fp8_dir), trust_remote_code)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    model.eval()

    pack = torch.load(fp8_dir / "fp8_scaled_weights.pt", map_location="cpu")
    fp8_state = pack["fp8_state"]
    scales = pack["scales"]
    schemes = pack["schemes"]
    passthrough = pack.get("passthrough", {})

    sd = {}
    for name, q in fp8_state.items():
        sd[name] = _dequant_fp8_to_dtype(q, scales[name], schemes.get(name, "per_tensor"), out_dtype)
    for name, t in passthrough.items():
        sd[name] = t

    model.load_state_dict(sd, strict=False)
    model.to(device)
    return model, tok

@torch.inference_mode()
def predict_choice_batched(model, tokenizer, prompt: str, device: torch.device) -> str:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    comp_ids = {ch: tokenizer(" " + ch, add_special_tokens=False).input_ids for ch in CHOICES}

    seqs = [prompt_ids + comp_ids[ch] for ch in CHOICES]
    max_len = max(len(s) for s in seqs)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((4, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((4, max_len), dtype=torch.long)

    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1

    input_ids = input_ids.to(device)
    attn = attn.to(device)

    out = model(input_ids=input_ids, attention_mask=attn)
    log_probs = torch.log_softmax(out.logits, dim=-1)

    start = len(prompt_ids)
    scores = {}
    for bi, ch in enumerate(CHOICES):
        ids = comp_ids[ch]
        lp = 0.0
        for j, tok_id in enumerate(ids):
            pos = start + j
            lp += float(log_probs[bi, pos - 1, tok_id].item())
        scores[ch] = lp

    return max(scores.items(), key=lambda kv: kv[1])[0]

def eval_mmlu(model, tok, mmlu_root: Path, shots: int, device: str, csv_workers: int, max_subjects: Optional[int]):
    subjects = list_subjects(mmlu_root)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    dev_map, test_map = preload_mmlu_csvs(mmlu_root, subjects, max_workers=csv_workers)

    dev = torch.device(device)
    per_subject = {}
    correct_total, total = 0, 0

    for subj in tqdm(subjects):
        dev_rows = dev_map[subj]
        test_rows = test_map[subj]
        c = 0
        for row in test_rows:
            prompt = build_fewshot_prompt(dev_rows, row, subj, shots)
            pred = predict_choice_batched(model, tok, prompt, dev)
            gold = row[5].strip()
            c += int(pred == gold)

        acc = c / max(1, len(test_rows))
        per_subject[subj] = acc
        correct_total += c
        total += len(test_rows)
        print(f"[{subj}] acc={acc:.4f} ({c}/{len(test_rows)})")

    macro = sum(per_subject.values()) / max(1, len(per_subject))
    micro = correct_total / max(1, total)
    return {
        "subjects": float(len(per_subject)),
        "macro_avg": float(macro),
        "micro_avg": float(micro),
        "n_total": float(total),
        "macro_points": float(macro * 100.0),
        "micro_points": float(micro * 100.0),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["original", "bf16_ft", "fp8_scaled"], required=True)
    ap.add_argument("--model_dir", type=str, help="for original/bf16_ft: local HF dir")
    ap.add_argument("--fp8_dir", type=str, help="for fp8_scaled: dir with fp8_scaled_weights.pt")
    ap.add_argument("--mmlu_root", type=str, required=True)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--csv_workers", type=int, default=8)
    ap.add_argument("--max_subjects", type=int, default=None)

    ap.add_argument("--original_size_dir", type=str, default=None, help="dir with ORIGINAL weights for ratio")
    ap.add_argument("--report_json", type=str, default=None)

    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()

    mmlu_root = Path(args.mmlu_root)
    assert (mmlu_root / "data" / "test").exists(), "mmlu_root must point to hendrycks/test repo"

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.mode in ("original", "bf16_ft"):
        if not args.model_dir:
            raise ValueError("--model_dir required for original/bf16_ft")
        tok = load_tokenizer_safe(args.model_dir, args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            device_map={"": args.device},
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()
        src = args.model_dir
    else:
        if not args.fp8_dir:
            raise ValueError("--fp8_dir required for fp8_scaled")
        model, tok = load_fp8_scaled_dequant_once(Path(args.fp8_dir), args.trust_remote_code, args.device, dtype)
        src = args.fp8_dir

    sizes = {"original_size_bytes": None, "compressed_size_bytes": None}
    if args.original_size_dir:
        sizes["original_size_bytes"] = int(weight_files_size_bytes(Path(args.original_size_dir)))
    if args.mode == "fp8_scaled":
        sizes["compressed_size_bytes"] = int(weight_files_size_bytes(Path(args.fp8_dir)))
    else:
        sizes["compressed_size_bytes"] = int(weight_files_size_bytes(Path(args.model_dir)))

    print(f"\n[model] {src}")
    if sizes["original_size_bytes"] is not None:
        print(f"  original_size_bytes: {sizes['original_size_bytes']/1e9:.3f} GB")
    print(f"  model_size_bytes:    {sizes['compressed_size_bytes']/1e9:.3f} GB")
    if sizes["original_size_bytes"]:
        print(f"  Compression_ratio:   {sizes['original_size_bytes']/sizes['compressed_size_bytes']:.3f}x")

    mmlu = eval_mmlu(model, tok, mmlu_root, args.shots, args.device, args.csv_workers, args.max_subjects)

    report = {"mode": args.mode, "source": src, "dtype": args.dtype, "sizes": sizes, "mmlu": mmlu}
    print("\n[result]\n" + json.dumps(report, indent=2))

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport saved: {args.report_json}")

if __name__ == "__main__":
    main()
