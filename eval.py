import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

CHOICES = ["A", "B", "C", "D"]
WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth"}


def load_tokenizer_safe(path_or_id: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(
            path_or_id,
            trust_remote_code=trust_remote_code,
            fix_mistral_regex=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            path_or_id,
            trust_remote_code=trust_remote_code,
        )


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

    dev_map: Dict[str, List[Tuple[str, str, str, str, str, str]]] = {}
    test_map: Dict[str, List[Tuple[str, str, str, str, str, str]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for subj in subjects:
            futures[("dev", subj)] = ex.submit(read_csv_rows, dev_dir / f"{subj}_dev.csv")
            futures[("test", subj)] = ex.submit(read_csv_rows, test_dir / f"{subj}_test.csv")

        for (kind, subj), fut in futures.items():
            rows = fut.result()
            if kind == "dev":
                dev_map[subj] = rows
            else:
                test_map[subj] = rows

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


def load_original(model_id_or_path: str, trust_remote_code: bool, device: str, load_dtype: torch.dtype):
    tok = load_tokenizer_safe(model_id_or_path, trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=load_dtype,
        device_map={"": device},
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def _dequant_fp8_to_dtype(q_fp8: torch.Tensor, scale: torch.Tensor, scheme: str, out_dtype: torch.dtype) -> torch.Tensor:
    q = q_fp8.to(out_dtype)

    if scheme == "per_row":
        if q.ndim == 2:
            s = scale.to(out_dtype).unsqueeze(1)  # [rows, 1]
            w = q * s
        else:
            w = q * scale.to(out_dtype)
    elif scheme == "per_tensor":
        w = q * scale.to(out_dtype)
    else:
        w = q

    return w.to(out_dtype)


def load_fp8_scaled_dequant_once(checkpoint_dir: Path, trust_remote_code: bool, device: str, out_dtype: torch.dtype):
    cfg = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=trust_remote_code)
    tok = load_tokenizer_safe(str(checkpoint_dir), trust_remote_code)

    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    model.eval()

    pack = torch.load(checkpoint_dir / "fp8_scaled_weights.pt", map_location="cpu")
    for k in ("fp8_state", "scales", "schemes"):
        if k not in pack:
            raise ValueError(f"Invalid fp8 pack: missing key '{k}' in {checkpoint_dir/'fp8_scaled_weights.pt'}")

    fp8_state: Dict[str, torch.Tensor] = pack["fp8_state"]
    scales: Dict[str, torch.Tensor] = pack["scales"]
    schemes: Dict[str, str] = pack["schemes"]
    passthrough: Dict[str, torch.Tensor] = pack.get("passthrough", {})

    sd: Dict[str, torch.Tensor] = {}
    for name, q in tqdm(fp8_state.items()):
        s = scales[name]
        scheme = schemes.get(name, "per_tensor")
        sd[name] = _dequant_fp8_to_dtype(q, s, scheme, out_dtype)

    for name, t in passthrough.items():
        sd[name] = t

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"missing keys: {len(missing)} (showing up to 20)\n  {missing[:20]}")
    if unexpected:
        print(f"unexpected keys: {len(unexpected)} (showing up to 20)\n  {unexpected[:20]}")

    model.to(device)
    return model, tok


def param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@torch.inference_mode()
def predict_choice_batched(model, tokenizer, prompt: str, device: torch.device) -> str:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

    comp_ids = {
        "A": tokenizer(" A", add_special_tokens=False).input_ids,
        "B": tokenizer(" B", add_special_tokens=False).input_ids,
        "C": tokenizer(" C", add_special_tokens=False).input_ids,
        "D": tokenizer(" D", add_special_tokens=False).input_ids,
    }

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
    logits = out.logits  # [B,T,V]
    log_probs = torch.log_softmax(logits, dim=-1)

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


def eval_mmlu(
    model,
    tok,
    mmlu_root: Path,
    shots: int,
    max_subjects: Optional[int],
    device: str,
    csv_workers: int,
) -> Dict[str, float]:
    subjects = list_subjects(mmlu_root)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    dev_map, test_map = preload_mmlu_csvs(mmlu_root, subjects, max_workers=csv_workers)
    dev = torch.device(device)

    per_subject = {}
    correct_total = 0
    total = 0

    for subj in subjects:
        print(f'Start processing {subj}')
        dev_rows = dev_map[subj]
        test_rows = test_map[subj]

        c = 0
        for row in tqdm(test_rows):
            prompt = build_fewshot_prompt(dev_rows, row, subj, shots)
            pred = predict_choice_batched(model, tok, prompt, dev)
            gold = row[5].strip()
            if pred == gold:
                c += 1

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


def load_report(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_score(original_report: Dict, compressed_report: Dict) -> Dict:
    orig_size = original_report["sizes"].get("original_size_bytes")
    comp_size = compressed_report["sizes"].get("compressed_size_bytes")
    if orig_size is None or comp_size is None or comp_size == 0:
        raise ValueError("Missing size bytes for score. Provide --original_size_dir and ensure compressed size is computed.")

    compression_ratio = orig_size / comp_size

    orig_metric = original_report["mmlu"]["macro_avg"]
    comp_metric = compressed_report["mmlu"]["macro_avg"]
    if orig_metric is None or orig_metric <= 0:
        raise ValueError("Original_metric is missing/zero; cannot compute Performance_drop.")

    performance_drop = (orig_metric - comp_metric) / orig_metric
    score = compression_ratio / (1.0 + performance_drop)

    return {
        "compression_ratio": float(compression_ratio),
        "performance_drop": float(performance_drop),
        "score": float(score),
        "original_metric_macro": float(orig_metric),
        "compressed_metric_macro": float(comp_metric),
        "original_size_bytes": int(orig_size),
        "compressed_size_bytes": int(comp_size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["original", "fp8_scaled"], required=True)
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--fp8_dir", type=str, default=None)

    ap.add_argument("--mmlu_root", type=str, required=True)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")  # CPU by default now
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--csv_workers", type=int, default=8)

    ap.add_argument("--report_json", type=str, default=None)
    ap.add_argument("--original_size_dir", type=str, default=None)
    ap.add_argument("--score_vs_report", type=str, default=None)

    ap.add_argument(
        "--load_dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="dtype for original mode",
    )
    ap.add_argument(
        "--fp8_dequant_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="dtype used after dequantization in fp8_scaled mode (on CPU).",
    )

    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_grad_enabled(False)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8)))
        torch.set_num_interop_threads(1)

    mmlu_root = Path(args.mmlu_root)
    assert (mmlu_root / "data" / "test").exists(), "mmlu_root must point to hendrycks/test repo containing data/dev and data/test"

    if args.mode == "original":
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.load_dtype]
        model, tok = load_original(args.model_id, args.trust_remote_code, args.device, dtype)
        src_label = args.model_id
    else:
        if not args.fp8_dir:
            raise ValueError("--fp8_dir is required for mode=fp8_scaled")
        dq_dtype = {"float16": torch.float16, "float32": torch.float32}[args.fp8_dequant_dtype]
        model, tok = load_fp8_scaled_dequant_once(Path(args.fp8_dir), args.trust_remote_code, args.device, dq_dtype)
        src_label = args.fp8_dir

    n_params = param_count(model)

    sizes = {
        "original_size_bytes": None,
        "compressed_size_bytes": None,
    }
    if args.original_size_dir:
        sizes["original_size_bytes"] = int(weight_files_size_bytes(Path(args.original_size_dir)))
    if args.mode == "fp8_scaled":
        sizes["compressed_size_bytes"] = int(weight_files_size_bytes(Path(args.fp8_dir)))

    print(f"\n[model] {src_label}")
    print(f"  params: {n_params:,}")
    if sizes["original_size_bytes"] is not None:
        print(f"  original_size_bytes(weights-only): {sizes['original_size_bytes']/1e9:.3f} GB")
    if sizes["compressed_size_bytes"] is not None:
        print(f"  compressed_size_bytes(weights-only): {sizes['compressed_size_bytes']/1e9:.3f} GB")
        if sizes["original_size_bytes"]:
            print(f"  Compression_ratio: {sizes['original_size_bytes']/sizes['compressed_size_bytes']:.3f}x")

    print("\n[mmlu] evaluating...")
    mmlu_res = eval_mmlu(
        model=model,
        tok=tok,
        mmlu_root=mmlu_root,
        shots=args.shots,
        max_subjects=args.max_subjects,
        device=args.device,
        csv_workers=args.csv_workers,
    )

    report = {
        "mode": args.mode,
        "source": src_label,
        "model_id": args.model_id,
        "shots": args.shots,
        "param_count": n_params,
        "sizes": sizes,
        "mmlu": mmlu_res,
    }

    print("\n[result]")
    print(json.dumps(report, indent=2))

    if args.score_vs_report:
        original_report = load_report(args.score_vs_report)
        report["score_eval"] = compute_score(original_report, report)
        print("\n[score]")
        print(json.dumps(report["score_eval"], indent=2))

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[report] saved: {args.report_json}")


if __name__ == "__main__":
    main()
