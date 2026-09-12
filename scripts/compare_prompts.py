# scripts/seek_prompts.py
# Quick sanity runner for Self-Ask *and* Direct prompts on two datasets:
# - First 5 items from GSM8k_200.jsonl (domain=math)
# - First 5 items from finqa_100.jsonl (domain=finance)
#
# It loads prompts from prompts.yaml and, per domain, runs up to two prompt types:
#   - domain="math":    type in {"selfask","direct"}
#   - domain="finance": type in {"selfask","direct"}
#
# Features:
# - Extracts compact final predictions (e.g., "60.3%", "162")
# - Compares with gold answers using 1% relative tolerance (numeric-aware)
# - Optional: --show_flow prints the FULL completion and a parsed "flow"
#   (sub-questions / intermediate answers) for Self-Ask outputs
# - Optional: --save_json path/to/file.json to dump raw prompt/completion logs
#
# Ollama endpoint per user: http://127.0.0.1:11434/api/generate
#
# Usage examples:
#   python scripts/seek_prompts.py --model mistral:7b-instruct --show_flow
#   python scripts/seek_prompts.py --gsm_path data/unified/GSM8k_200.jsonl --finqa_path data/unified/finqa_100.jsonl
#   python scripts/seek_prompts.py --temperature 0.2 --num_ctx 4096 --num_predict 512
#
# Notes:
# - Make sure your prompts end with a single final line 'Answer: <value>' to keep extraction stable.

import argparse
import json
import time
import yaml
import requests
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# -----------------------------
# Regex Helpers
# -----------------------------

SCALAR_RE = re.compile(r"(?xi)([-+])?\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%)?")
ANSWER_LINE_RE = re.compile(r"(?i)^\s*Answer:\s*(.+)$", flags=re.MULTILINE)

# Tolerant numeric parser used for correctness
NUM_RE = re.compile(
    r"""(?xi)
    ^\s*\$?\s*
    (?P<sign>[-+])?
    \s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
    \s*(?P<pct>%?)\s*$
    """
)

# Heuristics to parse "Self-Ask" style flow
FLOW_Q_RE = re.compile(r"(?i)^\s*(?:Q\d+|Question\s*\d+|Sub-?question\s*\d*|Step\s*\d+|\-|\*)[:\.\s]+(.+?)\s*$")
FLOW_A_RE = re.compile(r"(?i)^\s*(?:A\d+|Answer\s*\d+|\s*->\s*Answer|\s*Ans(?:wer)?|\s*Solution)\s*[:\.\s]+(.+?)\s*$")


# -----------------------------
# Core Utilities
# -----------------------------

def load_yaml(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def last_scalar_from_text(s: str) -> Optional[str]:
    m = list(SCALAR_RE.finditer(s))
    if not m:
        return None
    sign, num, pct = m[-1].groups()
    sign = sign or ""
    num = num.replace(",", "")
    pct = pct or ""
    return (sign + num + pct).strip().lstrip("$")

def extract_final_answer(text: str) -> str:
    """
    1) prefer last 'Answer:' line (first line after it)
    2) else last numeric token in candidate or full text
    3) else trimmed candidate
    """
    matches = list(ANSWER_LINE_RE.finditer(text))
    candidate = matches[-1].group(1).splitlines()[0].strip() if matches else text

    pred = last_scalar_from_text(candidate) or last_scalar_from_text(text)
    if pred:
        return pred
    return candidate.strip()

def to_numeric_token(s: str) -> Optional[Tuple[str, float]]:
    if s is None:
        return None
    s = str(s).strip()
    m = NUM_RE.match(s)
    if not m:
        return None
    sign = m.group("sign") or ""
    num = m.group("num").replace(",", "")
    pct = m.group("pct")
    val = float(f"{sign}{num}")
    if pct:
        return ("pct", val / 100.0)
    return ("abs", val)

def tol_equal(pred: str, gold: str, rel_tol: float = 0.01) -> bool:
    """
    Numeric-aware equality:
      - If both parse as numbers of same type (abs vs pct), check relative tolerance.
      - Else, fallback to normalized string compare.
    """
    p = to_numeric_token(pred)
    g = to_numeric_token(gold)
    if p and g and p[0] == g[0] and g[1] != 0:
        return abs(p[1] - g[1]) / abs(g[1]) <= rel_tol
    return (pred or "").strip().lower() == (gold or "").strip().lower()

def ollama_generate(model: str, prompt: str, options: Dict[str, Any]) -> Tuple[str, float]:
    url = "http://127.0.0.1:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=600)
    latency = time.time() - t0
    r.raise_for_status()
    out = r.json()
    return out["response"], latency


# -----------------------------
# Flow / Self-Ask Parsing
# -----------------------------

def parse_selfask_flow(text: str) -> Dict[str, List[str]]:
    """Best-effort parser to surface self-ask decomposition from completion."""
    questions: List[str] = []
    answers: List[str] = []
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Q-like patterns
        mq = FLOW_Q_RE.match(line_stripped)
        if mq:
            q = mq.group(1).strip()
            if q and q not in questions:
                questions.append(q)
            continue

        # A-like patterns
        ma = FLOW_A_RE.match(line_stripped)
        if ma:
            a = ma.group(1).strip()
            if a and a not in answers:
                answers.append(a)
            continue

        # Heuristic: lines ending with "?" (short) as sub-questions
        if line_stripped.endswith("?") and len(line_stripped) <= 180:
            if line_stripped not in questions:
                questions.append(line_stripped)

    return {"questions": questions, "answers": answers}


# -----------------------------
# Data Loading
# -----------------------------

def read_jsonl_first_n(path: str, n: int) -> List[Dict[str, Any]]:
    items = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            items.append(obj)
    return items

def normalize_record(obj: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Try to normalize different JSONL schemas into a (question, context, gold) triple.
    - For GSM8K: expect fields like 'question', 'answer' (scalar-like).
    - For FinQA: expect 'question', and 'answer' (numeric) or 'gold'/'final_result'.
                  For context, try 'context', 'evidence', or a concatenated string.
    """
    q = obj.get("question") or obj.get("query") or obj.get("Problem") or ""
    ctx = obj.get("context") or obj.get("passage") or obj.get("evidence") or ""
    if isinstance(ctx, (list, dict)):
        ctx = json.dumps(ctx, ensure_ascii=False)
    gold = obj.get("answer") or obj.get("gold") or obj.get("final_result") or ""
    gold = str(gold).strip()
    return {"question": str(q).strip(), "context": str(ctx).strip(), "gold": gold}


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts/prompts.yaml",
                    help="YAML file containing prompt templates.")
    ap.add_argument("--model", default="mistral:7b-instruct",
                    help="Ollama model name.")
    ap.add_argument("--gsm_path", default="data/unified/GSM8k_200.jsonl",
                    help="Path to GSM8K jsonl (first 5 items used).")
    ap.add_argument("--finqa_path", default="data/unified/finqa_100.jsonl",
                    help="Path to FinQA jsonl (first 5 items used).")
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--num_predict", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--show_flow", action="store_true",
                    help="If set, prints FULL completion and parsed flow (Self-Ask only).")
    ap.add_argument("--save_json", type=str, default="",
                    help="Optional path to save raw outputs as JSON (for debugging).")
    args = ap.parse_args()

    # Load prompts
    prompts = load_yaml(args.prompts)

    def find_prompt(domain_name: str, type_name: str) -> Optional[Dict[str,Any]]:
        return next((p for p in prompts
                     if str(p.get("domain","")).lower()==domain_name
                     and str(p.get("type","")).lower()==type_name
                     and "text" in p), None)

    # Pick Self-Ask and Direct prompts per domain (if present)
    math_selfask   = find_prompt("math", "selfask")
    math_direct    = find_prompt("math", "direct")
    fin_selfask    = find_prompt("finance", "selfask")
    fin_direct     = find_prompt("finance", "direct")

    if not math_selfask and not math_direct:
        print("WARNING: No math/selfask or math/direct prompts found; GSM8K items will be skipped.")
    if not fin_selfask and not fin_direct:
        print("WARNING: No finance/selfask or finance/direct prompts found; FinQA items will be skipped.")

    # Load first 5 items from each dataset
    gsm_items   = read_jsonl_first_n(args.gsm_path, 5) if (math_selfask or math_direct) else []
    finqa_items = read_jsonl_first_n(args.finqa_path, 5) if (fin_selfask or fin_direct) else []

    gsm_norm = [normalize_record(o, "math") for o in gsm_items]
    fin_norm = [normalize_record(o, "finance") for o in finqa_items]

    options = {
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    print("\n=== Running Self-Ask and Direct prompts on small mixed bench (GSM8K + FinQA) ===")
    print(f"Model: {args.model}")
    print(f"Decoding: num_ctx={args.num_ctx}, num_predict={args.num_predict}, "
          f"temperature={args.temperature}, seed={args.seed}\n")

    rows: List[Dict[str, Any]] = []
    raw_log: List[Dict[str, Any]] = []

    # Helper to run a batch with a specific prompt
    def run_batch(examples: List[Dict[str,Any]], prompt_obj: Dict[str,Any], domain_tag: str):
        for idx, ex in enumerate(examples, 1):
            tmpl = prompt_obj["text"]
            full_prompt = (tmpl.replace("{QUESTION}", ex["question"])
                                .replace("{CONTEXT}", ex.get("context","") or "(none)")
                                .replace("{NUMBERS_JSON}", "[]"))
            completion, latency = ollama_generate(args.model, full_prompt, options)
            pred = extract_final_answer(completion)
            ok = tol_equal(pred, ex["gold"], rel_tol=0.01)

            rid = f"{domain_tag}_{prompt_obj.get('type','prompt')}_{idx}"
            rows.append({
                "item_id": rid,
                "prompt_id": prompt_obj.get("id", f"{domain_tag}_{prompt_obj.get('type','prompt')}"),
                "domain": domain_tag,
                "ptype": prompt_obj.get("type",""),
                "pred": pred,
                "gold": ex["gold"],
                "correct_tol1pct": bool(ok),
                "latency_sec": round(latency, 3),
            })
            raw_log.append({
                "item_id": rid,
                "prompt_id": prompt_obj.get("id", f"{domain_tag}_{prompt_obj.get('type','prompt')}"),
                "domain": domain_tag,
                "ptype": prompt_obj.get("type",""),
                "prompt_text": full_prompt,
                "completion": completion,
                "pred": pred,
                "gold": ex["gold"],
                "latency_sec": latency
            })

            # Only show flow for self-ask prompts
            if args.show_flow and str(prompt_obj.get("type","")).lower() == "selfask":
                print("=" * 88)
                print(f"[{rid}] Prompt: {prompt_obj.get('id', f'{domain_tag}_selfask')}")
                print("-" * 88)
                print(">> FULL COMPLETION:")
                print(completion.strip())
                flow = parse_selfask_flow(completion)
                if flow["questions"] or flow["answers"]:
                    print("\n>> PARSED SELF-ASK FLOW (best-effort):")
                    if flow["questions"]:
                        print("  Sub-questions:")
                        for i, q in enumerate(flow["questions"], 1):
                            print(f"    Q{i}. {q}")
                    if flow["answers"]:
                        print("  Intermediate answers:")
                        for i, a in enumerate(flow["answers"], 1):
                            print(f"    A{i}. {a}")
                print("-" * 88)
                print(f"Extracted pred = {pred}   |   GOLD = {ex['gold']}   |   ok@1% = {ok}")
                print(f"Latency = {latency:.2f}s")
                print("=" * 88 + "\n")

    # Run GSM8K (math/selfask then math/direct if available)
    if gsm_norm:
        if math_selfask:
            print(f"Running GSM8K (first {len(gsm_norm)} items) with math/selfask: {math_selfask.get('id','math_selfask')}")
            run_batch(gsm_norm, math_selfask, "gsm8k")
        if math_direct:
            print(f"Running GSM8K (first {len(gsm_norm)} items) with math/direct: {math_direct.get('id','math_direct_prompt')}")
            run_batch(gsm_norm, math_direct, "gsm8k")

    # Run FinQA (finance/selfask then finance/direct if available)
    if fin_norm:
        if fin_selfask:
            print(f"\nRunning FinQA (first {len(fin_norm)} items) with finance/selfask: {fin_selfask.get('id','finance_selfask')}")
            run_batch(fin_norm, fin_selfask, "finqa")
        if fin_direct:
            print(f"Running FinQA (first {len(fin_norm)} items) with finance/direct: {fin_direct.get('id','finance_direct_prompt')}")
            run_batch(fin_norm, fin_direct, "finqa")

    # Pretty print compact table
    print("\n=== Summary (ok@1% tolerance) ===")
    colw = {"item_id": 20, "domain": 8, "ptype": 10, "prompt_id": 28, "pred": 16, "gold": 14, "ok": 8, "lat": 10}
    header = (f'{"item_id":<{colw["item_id"]}}  '
              f'{"domain":<{colw["domain"]}}  '
              f'{"ptype":<{colw["ptype"]}}  '
              f'{"prompt_id":<{colw["prompt_id"]}}  '
              f'{"pred":<{colw["pred"]}}  '
              f'{"gold":<{colw["gold"]}}  '
              f'{"ok@1%":<{colw["ok"]}}  '
              f'{"latency":<{colw["lat"]}}')
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f'{r["item_id"]:<{colw["item_id"]}}  '
              f'{r["domain"]:<{colw["domain"]}}  '
              f'{r["ptype"]:<{colw["ptype"]}}  '
              f'{r["prompt_id"]:<{colw["prompt_id"]}}  '
              f'{r["pred"]:<{colw["pred"]}}  '
              f'{r["gold"]:<{colw["gold"]}}  '
              f'{str(r["correct_tol1pct"]):<{colw["ok"]}}  '
              f'{r["latency_sec"]:<{colw["lat"]}}')

    # Optional JSON dump of raw results (prompt+completion) for debugging
    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({"model": args.model,
                       "decoding": options,
                       "runs": raw_log}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved raw outputs to: {outp}")

    print("\nDone. Ensure Self-Ask and Direct templates end with a single final line 'Answer: <value>' "
          "to keep extraction stable across prompt families.\n")


if __name__ == "__main__":
    main()
