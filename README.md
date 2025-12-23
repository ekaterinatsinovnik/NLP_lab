```bash
python compress.py \
  --model_id Qwen/Qwen3-8B \
  --local_model_dir ./qwen3_8b_original \
  --out_dir ./qwen3_8b_fp8_scaled \
  --trust_remote_code 
```
If data already acquired add --skip_download  

To run eval  
```bash
python eval.py \
  --mode fp8_scaled \
  --fp8_dir ./qwen3_8b_fp8_scaled \
  --mmlu_root ./hendrycks_test/ \
  --shots 1 \
  --device cpu | cuda \
  --fp8_dequant_dtype float16 \
  --trust_remote_code \
  --original_size_dir ./qwen3_8b_original \
  --report_json fp8_report.json
```

To run eval and compare  
```bash
python eval_mmlu.py \
  --mode fp8_scaled \
  --fp8_dir ./qwen3_8b_fp8_scaled \
  --model_id Qwen/Qwen3-8B \
  --mmlu_root ./test \
  --shots 1 \
  --device cuda \
  --trust_remote_code \
  --original_size_dir ./qwen3_8b_original \
  --score_vs_report original_report.json \
  --report_json fp8_report.json
```