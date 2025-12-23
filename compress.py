import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_tensor_fp8(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
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


def compress_state_dict_fp8_all(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    fp8_state: Dict[str, torch.Tensor] = {}
    scales: Dict[str, torch.Tensor] = {}
    schemes: Dict[str, str] = {}
    passthrough: Dict[str, torch.Tensor] = {}

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


def local_dir_has_weights(local_dir: Path) -> bool:
    if not local_dir.exists():
        return False
    exts = (".safetensors", ".bin", ".pt", ".pth")
    for p in local_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_size > 0:
            return True
    return any(p.is_file() and p.stat().st_size > 0 for p in local_dir.glob("*.safetensors"))


def cli_download_model(
    model_id: str,
    local_dir: Path,
    revision: Optional[str],
    token: Optional[str],
    resume: bool,
    force_download: bool,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["hf", "download", model_id, "--local-dir", str(local_dir)]
    if revision:
        cmd += ["--revision", revision]
    if resume:
        cmd += ["--resume-download"]
    if force_download:
        cmd += ["--force-download"]

    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGINGFACE_HUB_TOKEN"] = token
        cmd += ["--token", token]

    print("[hf-cli] running:")
    print("  " + " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip())

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"huggingface-cli download failed with exit code {rc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--local_model_dir", type=str, required=True)
    ap.add_argument("--hf_revision", type=str, default=None)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_download", action="store_true")
    ap.add_argument("--skip_download", action="store_true")

    ap.add_argument("--load_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_model_dir = Path(args.local_model_dir)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if args.skip_download:
        print("skip_download=1, will not call huggingface-cli.")
    elif local_dir_has_weights(local_model_dir):
        print(f"Found existing weight files in {local_model_dir}; skipping download.")
    else:
        cli_download_model(
            model_id=args.model_id,
            local_dir=local_model_dir,
            revision=args.hf_revision,
            token=token,
            resume=args.resume,
            force_download=args.force_download,
        )

    load_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.load_dtype]

    print(f"\n  loading model from local dir (CPU): {local_model_dir}")
    cfg = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=args.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=args.trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=load_dtype,
        device_map={"": "cpu"},
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=False,
    )
    model.eval()

    cfg.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    print("\n[compress] quantizing ALL weights to FP8 (storage)...")
    comp = compress_state_dict_fp8_all(model.state_dict())

    weights_path = out_dir / "fp8_scaled_weights.pt"
    meta_path = out_dir / "fp8_scaled_meta.json"

    torch.save(
        {
            "format": comp["format"],
            "fp8_state": comp["fp8_state"],
            "scales": comp["scales"],
            "schemes": comp["schemes"],
            "passthrough": comp["passthrough"],
        },
        weights_path,
    )

    n_params = int(sum(p.numel() for p in model.parameters()))
    theoretical_fp32_bytes = n_params * 4
    compressed_file_bytes = weights_path.stat().st_size

    meta = {
        "format": comp["format"],
        "model_id": args.model_id,
        "hf_revision": args.hf_revision,
        "local_model_dir": str(local_model_dir),
        "skip_download": bool(args.skip_download),
        "n_params": n_params,
        "theoretical_fp32_bytes": theoretical_fp32_bytes,
        "compressed_file_bytes": int(compressed_file_bytes),
        "compression_ratio_fp32_vs_fp8_file": float(theoretical_fp32_bytes / compressed_file_bytes)
        if compressed_file_bytes > 0 else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[done] saved compressed model to: {out_dir}")
    print(f"  fp8 weights file: {weights_path} ({compressed_file_bytes/1e9:.3f} GB)")
    print(f"  params: {n_params:,}")


if __name__ == "__main__":
    main()
