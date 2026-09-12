# peek_prompts3.py
# Runs prompts from prompts.yaml across GSM8K and FinQA with extended metrics.
# Metrics reported per dataset × prompt_id:
#   - avg relative numeric error
#   - EM@tau at 0.1%, 1%, 5%
#   - average LLM latency, Python program latency, total latency
#   - program error rate (Program/Hybrid only)
#   - average output tokens, average input tokens
#   - accuracy per 1k output tokens, per 1k input tokens, and per second
#
# Flows written to: flows_result[TIMESTAMP].json (includes question, prompt, completion, timings, tokens, code)

import argparse
import json
import time
import yaml
import requests
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

SCALAR_RE = re.compile(r"(?xi)([-+])?\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%)?")
ANSWER_LINE_RE = re.compile(r"(?i)^\s*Answer:\s*(.+)$", flags=re.MULTILINE)

NUM_RE = re.compile(
    r"""(?xi)
    ^\s*\$?\s*
    (?P<sign>[-+])?
    \s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
    \s*(?P<pct>%?)\s*$"""
)

FENCED_PY_RE = re.compile(r"```(?:python)?\s*(?P<code>[\s\S]*?)\s*```", flags=re.IGNORECASE)

UNSAFE_PATTERNS = [
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bfrom\s+os\b", r"\bfrom\s+sys\b", r"\bopen\s*\(", r"\beval\s*\(",
    r"\bexec\s*\(", r"\b__import__\s*\(", r"\brequests\b", r"\burllib\b",
    r"\bsocket\b", r"\bhttp\b", r"\bPopen\b",
]
SAFE_IMPORT_ALLOWLIST = ["math", "statistics", "re", "fractions", "decimal"]

def load_yaml(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

def normalize_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    q = obj.get("question") or obj.get("query") or obj.get("Problem") or ""
    ctx = obj.get("context") or obj.get("passage") or obj.get("evidence") or ""
    if isinstance(ctx, (list, dict)):
        ctx = json.dumps(ctx, ensure_ascii=False)
    gold = obj.get("answer") or obj.get("gold") or obj.get("final_result") or ""
    gold = str(gold).strip()
    return {"question": str(q).strip(), "context": str(ctx).strip(), "gold": gold}

def ollama_generate(model: str, prompt: str, options: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    url = "http://127.0.0.1:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=600)
    latency = time.time() - t0
    r.raise_for_status()
    out = r.json()
    out["_raw_latency"] = latency
    return out, latency

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
    matches = list(ANSWER_LINE_RE.finditer(text))
    candidate = matches[-1].group(1).splitlines()[0].strip() if matches else text
    pred = last_scalar_from_text(candidate) or last_scalar_from_text(text)
    return pred if pred else candidate.strip()

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
    try:
        val = float(f"{sign}{num}")
    except:
        return None
    if pct:
        return ("pct", val / 100.0)
    return ("abs", val)

def rel_close(pred: str, gold: str, rel_tol: float) -> bool:
    p = to_numeric_token(pred)
    g = to_numeric_token(gold)
    if p and g and p[0] == g[0]:
        if g[1] == 0.0:
            return abs(p[1]) <= rel_tol
        return abs(p[1] - g[1]) / max(1e-12, abs(g[1])) <= rel_tol
    # fall back to case-insensitive exact string match
    return (pred or "").strip().lower() == (gold or "").strip().lower()

def rel_error(pred: str, gold: str) -> Optional[float]:
    p = to_numeric_token(pred)
    g = to_numeric_token(gold)
    if p and g and p[0] == g[0]:
        if g[1] == 0.0:
            return abs(p[1])
        return abs(p[1] - g[1]) / abs(g[1])
    return None

def extract_all_python_blocks(completion: str) -> List[str]:
    return [m.group("code").strip() for m in FENCED_PY_RE.finditer(completion)]

def is_safe_python(code: str) -> bool:
    for pat in UNSAFE_PATTERNS:
        if re.search(pat, code):
            return False
    for m in re.finditer(r"(?i)^\s*import\s+([a-zA-Z0-9_]+)", code, flags=re.MULTILINE):
        if m.group(1) not in SAFE_IMPORT_ALLOWLIST:
            return False
    for m in re.finditer(r"(?i)^\s*from\s+([a-zA-Z0-9_]+)\s+import\s+", code, flags=re.MULTILINE):
        if m.group(1) not in SAFE_IMPORT_ALLOWLIST:
            return False
    return True

def run_python_code_sandboxed(code: str, timeout_sec: float = 6.0) -> Tuple[str, str, float, int]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            input=None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        rt = time.time() - t0
        return proc.stdout, proc.stderr, rt, proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"Timeout after {timeout_sec}s", timeout_sec, 124

def extract_answer_from_stdout(stdout: str) -> str:
    pred = last_scalar_from_text(stdout)
    if pred:
        return pred
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def infer_exec_kind_from_id(pid: str) -> str:
    lid = (pid or "").lower()
    if "hybrid" in lid:
        return "hybrid"
    if "program" in lid or "pot" in lid:
        return "program"
    return "text"

def build_prompt_text(prompt_text: str, question: str, context: str, sigfig: int, exec_kind: str) -> str:
    base = (prompt_text
                .replace("{QUESTION}", question)
                .replace("{CONTEXT}", context or "(none)")
                .replace("{NUMBERS_JSON}", "[]")
                .replace("{SIGFIG}", str(sigfig)))
    if exec_kind in ("program", "hybrid"):
        extra = f"\n\n[Formatting note] In your Python, print only the final numeric value. If numeric, round to {sigfig} significant figures."
        return base + extra
    else:
        extra = (
            f"\n\n[Formatting]\n"
            f"- If the answer is numeric, round to {sigfig} significant figures.\n"
            f"- End with a single line: Answer: <value>\n"
        )
        return base + extra

def rough_token_estimate(text: str) -> int:
    return max(1, len(text.strip().split()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts/prompts.yaml", help="YAML list of prompts with {id, text}.")
    ap.add_argument("--model", default="mistral:7b-instruct")
    ap.add_argument("--gsm_path", default="data/unified/GSM8k_200.jsonl")
    ap.add_argument("--finqa_path", default="data/unified/finqa_100.jsonl")
    ap.add_argument("--take_n", type=int, default=5)
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--num_predict", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sigfig", type=int, default=3)
    ap.add_argument("--save_dir", default=".")
    args = ap.parse_args()

    prompts = load_yaml(args.prompts)
    if not isinstance(prompts, list):
        raise ValueError("prompts.yaml must be a list of objects with 'id' and 'text'.")

    gsm_items = [normalize_record(o) for o in read_jsonl_first_n(args.gsm_path, args.take_n)]
    fin_items = [normalize_record(o) for o in read_jsonl_first_n(args.finqa_path, args.take_n)]

    options = {
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    flows_path = Path(args.save_dir) / f"flows_result{ts}.json"
    flows_log: List[Dict[str, Any]] = []

    buckets: Dict[Tuple[str,str], Dict[str, float]] = {}
    def ensure_bucket(ds: str, pid: str):
        if (ds, pid) not in buckets:
            buckets[(ds, pid)] = {
                "tot": 0,
                "em_0p1": 0,
                "em_1": 0,
                "em_5": 0,
                "rne_sum": 0.0,
                "rne_cnt": 0,
                "llm_lat_sum": 0.0,
                "py_lat_sum": 0.0,
                "tot_lat_sum": 0.0,
                "prog_attempts": 0,
                "prog_errors": 0,
                "out_tok_sum": 0,
                "in_tok_sum": 0,
            }

    for prompt_obj in prompts:
        pid = str(prompt_obj.get("id", "")).strip()
        ptext = str(prompt_obj.get("text", "")).strip()
        if not pid or not ptext:
            continue
        exec_kind = infer_exec_kind_from_id(pid)

        for ds_name, items in [("GSM8K", gsm_items), ("FinQA", fin_items)]:
            for idx, ex in enumerate(items, 1):
                ensure_bucket(ds_name, pid)
                full_prompt = build_prompt_text(ptext, ex["question"], ex.get("context",""), args.sigfig, exec_kind)
                out, llm_latency = ollama_generate(args.model, full_prompt, options)
                completion = out.get("response", "")
                pred_text = extract_final_answer(completion)

                out_tok = int(out.get("eval_count") or 0)
                in_tok  = int(out.get("prompt_eval_count") or 0)
                if out_tok <= 0:
                    out_tok = rough_token_estimate(completion)
                if in_tok <= 0:
                    in_tok = rough_token_estimate(full_prompt)

                py_out = ""
                py_err = ""
                py_latency = 0.0
                code_used = ""
                prog_attempt = 0
                prog_error = 0

                if exec_kind in ("program", "hybrid"):
                    prog_attempt = 1
                    code_blocks = extract_all_python_blocks(completion)
                    safe_blocks = [c for c in code_blocks if is_safe_python(c)]
                    if safe_blocks:
                        code_used = safe_blocks[-1]
                        py_out, py_err, py_latency, rc = run_python_code_sandboxed(code_used, timeout_sec=6.0)
                        if rc == 0:
                            pot_pred = extract_answer_from_stdout(py_out)
                            if pot_pred:
                                pred_text = pot_pred
                        else:
                            prog_error = 1
                    else:
                        prog_error = 1

                total_latency = llm_latency + py_latency

                em_0p1 = rel_close(pred_text, ex["gold"], 0.001)  # 0.1%
                em_1   = rel_close(pred_text, ex["gold"], 0.01)   # 1%
                em_5   = rel_close(pred_text, ex["gold"], 0.05)   # 5%
                rne = rel_error(pred_text, ex["gold"])

                b = buckets[(ds_name, pid)]
                b["tot"] += 1
                b["em_0p1"] += int(bool(em_0p1))
                b["em_1"]   += int(bool(em_1))
                b["em_5"]   += int(bool(em_5))
                if rne is not None:
                    b["rne_sum"] += float(rne)
                    b["rne_cnt"] += 1
                b["llm_lat_sum"]  += llm_latency
                b["py_lat_sum"]   += py_latency
                b["tot_lat_sum"]  += total_latency
                b["prog_attempts"] += prog_attempt
                b["prog_errors"]   += prog_error
                b["out_tok_sum"]   += out_tok
                b["in_tok_sum"]    += in_tok

                flows_log.append({
                    "dataset": ds_name,
                    "prompt_id": pid,
                    "item_index": idx,
                    "question": ex["question"],
                    "context": ex.get("context",""),
                    "gold": ex["gold"],
                    "prompt_text": full_prompt,
                    "completion": completion,
                    "pred": pred_text,
                    "em_0p1": bool(em_0p1),
                    "em_1": bool(em_1),
                    "em_5": bool(em_5),
                    "rne": rne,
                    "llm_latency": llm_latency,
                    "py_latency": py_latency,
                    "total_latency": total_latency,
                    "out_tokens": out_tok,
                    "in_tokens": in_tok,
                    "exec_kind": exec_kind,
                    "python_code": code_used if code_used else None,
                    "python_stdout": py_out if py_out else None,
                    "python_stderr": py_err if py_err else None,
                    "program_error": bool(prog_error) if exec_kind in ("program","hybrid") else None,
                })

    flows_path.parent.mkdir(parents=True, exist_ok=True)
    with open(flows_path, "w", encoding="utf-8") as f:
        json.dump(flows_log, f, ensure_ascii=False, indent=2)

    def pct(x, n):
        return f"{(100.0 * x / n):.1f}%" if n > 0 else "n/a"
    def avg(x, n):
        return (x / n) if n > 0 else 0.0

    print("\n=== Summary Metrics by Dataset × PromptID ===")
    wds, wpid = 8, 30
    header = (
        f'{"dataset":<{wds}}  {"prompt_id":<{wpid}}  '
        f'{"EM@0.1%":>8}  {"EM@1%":>8}  {"EM@5%":>8}  '
        f'{"avg RNE":>10}  '
        f'{"LLM s":>8}  {"Py s":>8}  {"Total s":>8}  '
        f'{"ProgErr%":>8}  '
        f'{"outTok":>8}  {"inTok":>8}  '
        f'{"Acc/1kOut":>10}  {"Acc/1kIn":>10}  {"Acc/sec":>8}'
    )
    print(header)
    print("-" * len(header))

    ordered_keys = sorted(buckets.keys(), key=lambda k: (k[0], k[1]))
    for ds, pid in ordered_keys:
        b = buckets[(ds, pid)]
        tot = int(b["tot"]) or 1
        em1 = b["em_1"]
        em_rate_0p1 = pct(b["em_0p1"], tot)
        em_rate_1   = pct(em1, tot)
        em_rate_5   = pct(b["em_5"], tot)

        avg_rne = avg(b["rne_sum"], b["rne_cnt"]) if b["rne_cnt"] > 0 else float('nan')
        llm_s = avg(b["llm_lat_sum"], tot)
        py_s  = avg(b["py_lat_sum"], tot)
        tot_s = avg(b["tot_lat_sum"], tot)
        prog_err_pct = (100.0 * b["prog_errors"] / b["prog_attempts"]) if b["prog_attempts"] > 0 else 0.0

        avg_out_tok = avg(b["out_tok_sum"], tot)
        avg_in_tok  = avg(b["in_tok_sum"], tot)

        acc_per_1k_out = (100.0 * em1 / tot) / max(1.0, avg_out_tok / 1000.0)
        acc_per_1k_in  = (100.0 * em1 / tot) / max(1.0, avg_in_tok / 1000.0)
        acc_per_sec    = (100.0 * em1 / tot) / max(1e-6, tot_s)

        print(
            f'{ds:<{wds}}  {pid:<{wpid}}  '
            f'{em_rate_0p1:>8}  {em_rate_1:>8}  {em_rate_5:>8}  '
            f'{avg_rne:>10.4f}  '
            f'{llm_s:>8.2f}  {py_s:>8.2f}  {tot_s:>8.2f}  '
            f'{prog_err_pct:>8.1f}  '
            f'{int(avg_out_tok):>8}  {int(avg_in_tok):>8}  '
            f'{acc_per_1k_out:>10.2f}  {acc_per_1k_in:>10.2f}  {acc_per_sec:>8.2f}'
        )

    print(f"\nFlows written to: {flows_path}")

if __name__ == "__main__":
    main()
