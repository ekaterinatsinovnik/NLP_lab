### Results:
#### Models sizes:
   
   "original_size_bytes": 16381516776

   "compressed_size_bytes": 8194399304

   (1.999x compress ratio)


#### MMLU metrics:
  - original model 
    "macro_avg": 0.7543613921076655
    "micro_avg": 0.731448511608033

  - compressed
    "macro_avg": 0.7533491826228992
    "micro_avg": 0.7307363623415468

  - finetuned
    "macro_avg": 0.7533774755910347
    "micro_avg": 0.7305227175616009

### Download dataset
```bash
git clone https://github.com/hendrycks/test.git 
cd test
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
export PATH_TO_HENDRYCKS_TEST=$(pwd)
```

### Compress
```bash
python compress.py \
  --model_id Qwen/Qwen3-8B \
  --local_model_dir ./qwen3_8b_original \
  --out_dir ./qwen3_8b_fp8_scaled \
  --trust_remote_code 
```
If data already acquired add `--skip_download`  

### Eval  
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


### Finetune
```bash
python train.py \
  --fp8_dir ./qwen3_8b_fp8_scaled \
  --mmlu_root ${PATH_TO_HENDRYCKS_TEST} \
  --out_dir ./qwen3_8b_ft_cpu \
  --trust_remote_code \
  --device cuda \
  --epochs 1 \
  --bf16 \
  --lr 1e-5 \
  --per_device_batch_size 8 \
  --grad_accum 8 \
  --max_length 512

```
to recompress after fine-tuning add `--recompress_fp8 --out_fp8_dir ./qwen3_8b_ft_fp8`  


### Eval after recompressing:
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