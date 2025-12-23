Download dataset
```bash
git clone https://github.com/hendrycks/test.git 
cd test
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
export PATH_TO_HENDRYCKS_TEST=$(pwd)
```

```bash
python compress.py \
  --model_id Qwen/Qwen3-8B \
  --local_model_dir ./qwen3_8b_original \
  --out_dir ./qwen3_8b_fp8_scaled \
  --trust_remote_code 
```
If data already acquired add `--skip_download`  

To run eval  
```bash
python eval.py \
  --mode fp8_scaled \
  --fp8_dir ./qwen3_8b_fp8_scaled \
  --mmlu_root ${PATH_TO_HENDRYCKS_TEST} \
  --shots 1 \
  --device cpu | cuda \
  --fp8_dequant_dtype float16 \
  --trust_remote_code \
  --original_size_dir ./qwen3_8b_original \
  --report_json fp8_report.json
```

To run eval and compare  
```bash
python eval.py \
  --mode fp8_scaled \
  --fp8_dir ./qwen3_8b_fp8_scaled \
  --model_id Qwen/Qwen3-8B \
  --mmlu_root ${PATH_TO_HENDRYCKS_TEST} \
  --shots 1 \
  --device cuda \
  --trust_remote_code \
  --original_size_dir ./qwen3_8b_original \
  --score_vs_report original_report.json \
  --report_json fp8_report.json
```
finetune
```bash
python train.py \
  --fp8_dir ./qwen3_8b_fp8_scaled \
  --mmlu_root ${PATH_TO_HENDRYCKS_TEST} \
  --out_dir ./qwen3_8b_ft_cpu \
  --trust_remote_code \
  --device cpu | cuda \
  --epochs 1 \
  --bf16 \
  --lr 1e-5 \
  --per_device_batch_size 1 \
  --grad_accum 8 \
  --max_length 512

```
to recompress after fine-tuning add `--recompress_fp8 --out_fp8_dir ./qwen3_8b_ft_fp8`  

inference
```bash
python inference.py \
  --mode bf16_ft \
  --model_dir ./qwen3_8b_ft_bf16 \
  --mmlu_root ${PATH_TO_HENDRYCKS_TEST} \
  --shots 1 \
  --device cuda \
  --dtype bf16 \
  --trust_remote_code \
  --original_size_dir ./qwen3_8b_original \
  --report_json ft_bf16_report.json
```
infer after recompressing:
```bash
python inference.py \
  --mode fp8_scaled \
  --fp8_dir ./qwen3_8b_ft_fp8 \
  --mmlu_root ${PATH_TO_HENDRYCKS_TEST} \
  --shots 1 \
  --device cuda \
  --dtype bf16 \
  --trust_remote_code \
  --original_size_dir ./qwen3_8b_original \
  --report_json ft_fp8_report.json
```