import argparse, json, time, yaml, requests, pandas as pd, re
from pathlib import Path
from rich.progress import track
import re

def load_yaml(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

def read_jsonl(p):
    for line in open(p, "r", encoding="utf-8"):
        if line.strip():
            yield json.loads(line)

def pick_prompt(prompts, domain_hint, kind_preference=("program","selfask")):
    # prefer domain-matched; else fallback "any"
    for kind in kind_preference:
        for p in prompts:
            if (p["domain"] == domain_hint or p["domain"] == "any") and p["type"] == kind:
                return p
    return prompts[0]

def fill_prompt(tmpl, q, ctx, numbers_json):
    return (tmpl.replace("{QUESTION}", q)
                .replace("{CONTEXT}", ctx if ctx else "(none)")
                .replace("{NUMBERS_JSON}", json.dumps(numbers_json, ensure_ascii=False)))

def extract_final_answer(text: str):
    # 1) Prefer the **last** occurrence of 'Answer:' (final line convention)
    matches = list(re.finditer(r"Answer:\s*(.+)", text, flags=re.IGNORECASE))
    if matches:
        candidate = matches[-1].group(1).strip()
        # stop at the first hard break
        candidate = candidate.splitlines()[0].strip()

        # 2) Normalize common finance tokens like "[in thousands]" annotations
        candidate = re.sub(r"\[.*?in (thousands|millions).*?\]", "", candidate, flags=re.I).strip()

        # 3) If the candidate still looks like a sentence, try to extract the last scalar in it
        scalars = re.findall(r"[-+]?\d[\d,]*\.?\d*\s*%?", candidate)
        if scalars:
            return scalars[-1].strip().lstrip("$")
        return candidate

    # 4) Fallback: last scalar anywhere
    scalars = re.findall(r"[-+]?\d[\d,]*\.?\d*\s*%?", text)
    return scalars[-1].strip().lstrip("$") if scalars else ""

def ollama_generate(model, prompt, options):
    url = "http://127.0.0.1:11435/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=300)
    latency = time.time() - t0
    r.raise_for_status()
    out = r.json()
    return out["response"], latency, out

def run(config_path, prompts_path):
    cfg = load_yaml(config_path)
    prompts = load_yaml(prompts_path)

    rows = []
    for input_path in cfg["io"]["inputs"]:
        for ex in track(read_jsonl(input_path), description=f"Eval {Path(input_path).name}"):
            domain = ex.get("domain","any")
            # choose template based on domain (for cross-domain, swap order later)
            pmeta = pick_prompt(prompts, domain)
            full_prompt = fill_prompt(pmeta["text"], ex["question"], ex.get("context",""), ex.get("numbers_json", []))

            completion, lat, meta = ollama_generate(cfg["model"], full_prompt, cfg["options"])
            rows.append({
                "experiment": cfg["experiment_name"],
                "model": cfg["model"],
                "dataset": ex.get("dataset",""),
                "item_id": ex.get("id",""),
                "domain_prompt": pmeta["domain"],
                "type_prompt": pmeta["type"],
                "prompt_id": pmeta["id"],
                "question": ex["question"],
                "gold": ex.get("answer",""),
                "completion": completion,
                "pred": extract_final_answer(completion),
                "latency_sec": lat,
                "eval_count": meta.get("eval_count", None),
                "num_ctx": cfg["options"]["num_ctx"],
                "num_predict": cfg["options"]["num_predict"],
                "temperature": cfg["options"]["temperature"],
                "seed": cfg["options"]["seed"]
            })

    df = pd.DataFrame(rows)
    out_csv = Path(cfg["io"]["out_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_config.yaml")
    ap.add_argument("--prompts", default="prompts/prompts.yaml")
    args = ap.parse_args()
    run(args.config, args.prompts)
