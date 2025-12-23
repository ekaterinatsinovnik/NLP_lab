import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_tokenizer_safe(path_or_id: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(path_or_id, trust_remote_code=trust_remote_code, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(path_or_id, trust_remote_code=trust_remote_code)


def list_subjects_dev(mmlu_root: Path) -> List[str]:
    dev_dir = mmlu_root / "data" / "dev"
    return sorted([p.name.replace("_dev.csv", "") for p in dev_dir.glob("*_dev.csv")])

def read_csv_rows(path: Path) -> List[Tuple[str, str, str, str, str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) < 6:
                continue
            rows.append((r[0], r[1], r[2], r[3], r[4], r[5]))
    return rows

def subject_to_title(subj: str) -> str:
    return subj.replace("_", " ")

def build_prompt(subj: str, q: str, a: str, b: str, c: str, d: str) -> str:
    return (
        f"The following is a multiple choice question about {subject_to_title(subj)}.\n\n"
        f"Question: {q}\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n"
        f"Answer:"
    )


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

def dequant_fp8_pack_to_dtype(pack: Dict, out_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    fp8_state = pack["fp8_state"]
    scales = pack["scales"]
    schemes = pack["schemes"]
    passthrough = pack.get("passthrough", {})

    sd: Dict[str, torch.Tensor] = {}
    for name, q in fp8_state.items():
        q32 = q.to(out_dtype)
        s = scales[name].to(out_dtype)
        scheme = schemes.get(name, "per_tensor")

        if scheme == "per_row" and q32.ndim == 2:
            w = q32 * s.unsqueeze(1)
        elif scheme == "per_tensor":
            w = q32 * s
        else:
            w = q32

        sd[name] = w.to(out_dtype)

    for name, t in passthrough.items():
        sd[name] = t
    return sd

def quantize_tensor_fp8(t: torch.Tensor):
    t = t.detach().to(torch.float32).cpu()
    if t.numel() == 0:
        scale = torch.tensor(1.0, dtype=torch.float16)
        return t.to(FP8_DTYPE), scale, "empty"
    if t.ndim == 2:
        max_abs = t.abs().amax(dim=1)
        scale = (max_abs / FP8_MAX).clamp_min(1e-8).to(torch.float16)
        q = (t / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        return q, scale, "per_row"
    else:
        max_abs = t.abs().max()
        scale_val = float(max_abs / FP8_MAX) if max_abs > 0 else 1.0
        scale_val = max(scale_val, 1e-8)
        scale = torch.tensor(scale_val, dtype=torch.float16)
        q = (t / scale_val).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        return q, scale, "per_tensor"

def compress_state_dict_fp8_all(state: Dict[str, torch.Tensor]) -> Dict:
    fp8_state, scales, schemes, passthrough = {}, {}, {}, {}
    for name, t in state.items():
        if not torch.is_tensor(t):
            continue
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            passthrough[name] = t.cpu()
            continue
        q, s, scheme = quantize_tensor_fp8(t)
        fp8_state[name] = q
        scales[name] = s
        schemes[name] = scheme
    return {
        "format": "fp8_e4m3fn_scaled_v3_all",
        "fp8_state": fp8_state,
        "scales": scales,
        "schemes": schemes,
        "passthrough": passthrough,
    }


class MMLUAnswerOnlyDataset(Dataset):
    def __init__(self, mmlu_root: Path, tokenizer, max_length: int, max_subjects: Optional[int] = None):
        self.tok = tokenizer
        self.samples = []

        dev_dir = mmlu_root / "data" / "dev"
        subjects = list_subjects_dev(mmlu_root)
        if max_subjects is not None:
            subjects = subjects[:max_subjects]

        for subj in subjects:
            rows = read_csv_rows(dev_dir / f"{subj}_dev.csv")
            for (q, a, b, c, d, ans) in rows:
                ans = ans.strip()
                prompt = build_prompt(subj, q, a, b, c, d)
                completion = " " + ans

                prompt_ids = self.tok(prompt, add_special_tokens=False).input_ids
                full_ids = self.tok(prompt + completion, add_special_tokens=False).input_ids

                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

                if len(full_ids) > max_length:
                    overflow = len(full_ids) - max_length
                    full_ids = full_ids[overflow:]
                    labels = labels[overflow:]

                attn = [1] * len(full_ids)

                self.samples.append(
                    {
                        "input_ids": torch.tensor(full_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(attn, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class PadCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, features):
        max_len = max(f["input_ids"].shape[0] for f in features)
        pad_id = self.tok.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            L = f["input_ids"].shape[0]
            pad = max_len - L
            input_ids.append(torch.cat([f["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([f["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))
            labels.append(torch.cat([f["labels"], torch.full((pad,), -100, dtype=torch.long)]))

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

def resolve_train_dtype(device: str, bf16_flag: bool, fp16_flag: bool) -> torch.dtype:
    if device == "cuda":
        if fp16_flag and not bf16_flag:
            return torch.float16
        return torch.bfloat16
    if bf16_flag:
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16)
            return torch.bfloat16
        except Exception:
            return torch.float32
    return torch.float32

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8_dir", type=str, required=True)
    ap.add_argument("--mmlu_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_subjects", type=int, default=None)

    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--per_device_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--bf16", action="store_true", help="prefer bf16")
    ap.add_argument("--fp16", action="store_true", help="prefer fp16")

    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)

    ap.add_argument("--recompress_fp8", action="store_true")
    ap.add_argument("--out_fp8_dir", type=str, default=None)

    args = ap.parse_args()
    set_seed(args.seed)

    fp8_dir = Path(args.fp8_dir)
    mmlu_root = Path(args.mmlu_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert (mmlu_root / "data" / "dev").exists(), "mmlu_root must have data/dev"
    assert (fp8_dir / "fp8_scaled_weights.pt").exists(), "fp8_scaled_weights.pt not found in fp8_dir"

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")

    # CPU tuning (optional)
    if args.device == "cpu":
        torch.set_grad_enabled(True)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8)))
        torch.set_num_interop_threads(1)

    # Load config/tokenizer
    cfg = AutoConfig.from_pretrained(fp8_dir, trust_remote_code=args.trust_remote_code)
    tok = load_tokenizer_safe(str(fp8_dir), args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Create model
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=args.trust_remote_code)

    # Dequantize into training dtype
    pack = torch.load(fp8_dir / "fp8_scaled_weights.pt", map_location="cpu")
    train_dtype = resolve_train_dtype(args.device, args.bf16, args.fp16)
    sd = dequant_fp8_pack_to_dtype(pack, out_dtype=train_dtype)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing up to 20)\n  {missing[:20]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing up to 20)\n  {unexpected[:20]}")

    model.to(args.device)
    model.train()

    # Dataset
    train_ds = MMLUAnswerOnlyDataset(
        mmlu_root=mmlu_root,
        tokenizer=tok,
        max_length=args.max_length,
        max_subjects=args.max_subjects,
    )
    collator = PadCollator(tok)

    use_bf16 = (args.device == "cuda") and (train_dtype == torch.bfloat16)
    use_fp16 = (args.device == "cuda") and (train_dtype == torch.float16)

    grad_ckpt = True if args.device == "cuda" else False

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=4 if args.device == "cuda" else 0,
        gradient_checkpointing=grad_ckpt,
        torch_compile=False,
        use_cpu=args.device != "cuda",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    cfg.save_pretrained(out_dir)

    print(f"[done] fine-tuned model saved to: {out_dir}")

    if args.recompress_fp8:
        if not args.out_fp8_dir:
            raise ValueError("--out_fp8_dir required when --recompress_fp8 is set")

        out_fp8 = Path(args.out_fp8_dir)
        out_fp8.mkdir(parents=True, exist_ok=True)

        print("[fp8] recompressing fine-tuned model to storage-FP8...")
        comp = compress_state_dict_fp8_all(model.state_dict())

        torch.save(
            {
                "format": comp["format"],
                "fp8_state": comp["fp8_state"],
                "scales": comp["scales"],
                "schemes": comp["schemes"],
                "passthrough": comp["passthrough"],
            },
            out_fp8 / "fp8_scaled_weights.pt",
        )
        cfg.save_pretrained(out_fp8)
        tok.save_pretrained(out_fp8)

        meta = {
            "base_fp8_dir": str(fp8_dir),
            "fine_tuned_dir": str(out_dir),
            "format": comp["format"],
        }
        (out_fp8 / "fp8_scaled_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[done] re-compressed FP8 checkpoint saved to: {out_fp8}")


if __name__ == "__main__":
    main()
