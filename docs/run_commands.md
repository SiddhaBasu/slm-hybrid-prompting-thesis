# 1) pull data & unify
python .\scripts\download_datasets.py
python .\scripts\unify_datasets.py

# 2) run on Mistral-7B Instruct
python .\scripts\run_ollama_eval.py --config configs\run_config.yaml --prompts prompts\prompts.yaml
python .\scripts\eval_metrics.py .\outputs\slm_results.csv

# 3) switch model in configs/run_config.yaml
# 4) flip cross-domain override (see step 8), repeat
# 5) add hybrid prompt in prompts.yaml, repeat
