# peek_prompts2.py
# Runs prompts from prompts1.yaml across GSM8K and FinQA.
# - Removes "style" as an identifier; uses only prompt ID for reporting and aggregation.
# - Implements per-item summary table (similar to original peek_prompts.py).
# - Exact Match (EM) uses string-equality OR a very tight numeric match at 0.01% rel. error to avoid sig-fig false negatives.
# - Relative Numeric Match (RNM) uses a configurable tolerance (default 5%).
# - Program-of-Thought and Hybrid prompts execute Python; others are text-only.
# - Writes detailed flows to flows_result[TIMESTAMP].json including the question.

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

# -----------------------------
# Regex and parsing helpers
# -----------------------------

SCALAR_RE = re.compile(r"(?xi)([-+])?\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%)?")
ANSWER_LINE_RE = re.compile(r"(?i)^\s*Answer:\s*(.+)$", flags=re.MULTILINE)

NUM_RE = re.compile(
    r"""(?xi)
    ^\s*\$?\s*
    (?P<sign>[-+])?
    \s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
    \s*(?P<pct>%?)\s*$
    """
)

FLOW_Q_RE = re.compile(r"(?i)^\s*(?:Q\d+|Question\s*\d+|Sub-?question\s*\d*|Step\s*\d+|\-|\*)[:\.\s]+(.+?)\s*$")
FLOW_A_RE = re.compile(r"(?i)^\s*(?:A\d+|Answer\s*\d+|\s*->\s*Answer|\s*Ans(?:wer)?|\s*Solution)\s*[:\.\s]+(.+?)\s*$")

FENCED_PY_RE = re.compile(r"```(?:python)?\s*(?P<code>[\s\S]*?)\s*```", flags=re.IGNORECASE)
NUM_TOKEN_RE = re.compile(r"(?<![A-Za-z_])(-?\d+(?:\.\d+)?)")

UNSAFE_PATTERNS = [
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bfrom\s+os\b", r"\bfrom\s+sys\b", r"\bopen\s*\(", r"\beval\s*\(",
    r"\bexec\s*\(", r"\b__import__\s*\(", r"\brequests\b", r"\burllib\b",
    r"\bsocket\b", r"\bhttp\b", r"\bPopen\b",
]
SAFE_IMPORT_ALLOWLIST = ["math", "statistics", "re", "fractions", "decimal"]

# -----------------------------
# IO and HTTP
# -----------------------------

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
# Parsing predictions and metrics
# -----------------------------

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
    val = float(f"{sign}{num}")
    if pct:
        return ("pct", val / 100.0)
    return ("abs", val)

def exact_match_relaxed(pred: str, gold: str, rel_tol_em: float = 0.0001) -> bool:
    """
    Exact Match with a relaxed numeric path:
    - Return True if strings match case-insensitively.
    - Else, if both parse as numeric of same type, accept if within rel_tol_em (default 0.01%).
    """
    if (pred or "").strip().lower() == (gold or "").strip().lower():
        return True
    p = to_numeric_token(pred)
    g = to_numeric_token(gold)
    if p and g and p[0] == g[0]:
        if g[1] == 0:
            return abs(p[1]) <= rel_tol_em  # treat near-zero as match
        return abs(p[1] - g[1]) / abs(g[1]) <= rel_tol_em
    return False

def relative_numeric_match(pred: str, gold: str, rel_tol: float) -> bool:
    p = to_numeric_token(pred)
    g = to_numeric_token(gold)
    if p and g and p[0] == g[0] and g[1] != 0:
        return abs(p[1] - g[1]) / abs(g[1]) <= rel_tol
    return False

# -----------------------------
# Self-Ask flow parsing saved to file
# -----------------------------

def parse_selfask_flow(text: str) -> Dict[str, List[str]]:
    questions: List[str] = []
    answers: List[str] = []
    for line in text.splitlines():
        ls = line.strip()
        if not ls:
            continue
        mq = FLOW_Q_RE.match(ls)
        if mq:
            q = mq.group(1).strip()
            if q and q not in questions:
                questions.append(q)
            continue
        ma = FLOW_A_RE.match(ls)
        if ma:
            a = ma.group(1).strip()
            if a and a not in answers:
                answers.append(a)
            continue
        if ls.endswith("?") and len(ls) <= 180:
            if ls not in questions:
                questions.append(ls)
    return {"questions": questions, "answers": answers}

# -----------------------------
# PoT code selection and execution
# -----------------------------

def extract_all_python_blocks(completion: str) -> List[Tuple[int, int, str]]:
    blocks: List[Tuple[int,int,str]] = []
    for m in FENCED_PY_RE.finditer(completion):
        start, end = m.span()
        code = m.group("code").strip()
        blocks.append((start, end, code))
    return blocks

def is_safe_python(code: str) -> bool:
    for pat in UNSAFE_PATTERNS:
        if re.search(pat, code):
            return False
    for m in re.finditer(r"(?i)^\s*import\s+([a-zA-Z0-9_]+)", code, flags=re.MULTILINE):
        mod = m.group(1)
        if mod not in SAFE_IMPORT_ALLOWLIST:
            return False
    for m in re.finditer(r"(?i)^\s*from\s+([a-zA-Z0-9_]+)\s+import\s+", code, flags=re.MULTILINE):
        mod = m.group(1)
        if mod not in SAFE_IMPORT_ALLOWLIST:
            return False
    return True

def numbers_in_text(s: str) -> List[str]:
    return [m.group(1) for m in NUM_TOKEN_RE.finditer(s or "")]

def score_code_block(code: str, q_numbers: List[str]) -> float:
    score = 0.0
    if re.search(r"\bprint\s*\(", code):
        score += 2.0
    def to_float(x):
        try:
            return float(x.replace(",", ""))
        except:
            return None
    q_vals = [to_float(x) for x in q_numbers]
    code_vals = [to_float(x) for x in numbers_in_text(code)]
    for qv in q_vals:
        if qv is None:
            continue
        for cv in code_vals:
            if cv is None:
                continue
            if abs(qv - cv) <= max(1e-9, 1e-3 * max(1.0, abs(qv))):
                score += 1.0
                break
    if re.search(r"\bTODO\b|\bpass\b|NotImplementedError", code):
        score -= 1.0
    return score

def select_python_block_for_question(completion: str, question: str) -> Optional[str]:
    blocks = extract_all_python_blocks(completion)
    if not blocks:
        return None
    q_nums = numbers_in_text(question)
    scored = []
    for (start, end, code) in blocks:
        s = score_code_block(code, q_nums)
        scored.append((s, start, code))
    scored.sort(key=lambda t: (t[0], t[1]))
    best = scored[-1]
    return best[2] if best[0] > 0.0 else blocks[-1][2]

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

# -----------------------------
# Prompt handling
# -----------------------------

def infer_exec_kind_from_id(pid: str) -> str:
    """
    Internal execution decision only. Not shown in outputs.
    Returns one of {"program","hybrid","text"} where text = direct/selfask/other.
    """
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

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts/prompts1.yaml", help="YAML file with prompt templates (list of {id, text}).")
    ap.add_argument("--model", default="granite3.3:8b", help="Ollama model name.")
    ap.add_argument("--gsm_path", default="data/unified/GSM8k_200.jsonl", help="Path to GSM8K jsonl.")
    ap.add_argument("--finqa_path", default="data/unified/finqa_100.jsonl", help="Path to FinQA jsonl.")
    ap.add_argument("--take_n", type=int, default=5, help="Number of items from each dataset.")
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--num_predict", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sigfig", type=int, default=3)
    ap.add_argument("--rel_tol", type=float, default=0.05, help="Relative tolerance for RNM, e.g., 0.05 for 5%.")
    ap.add_argument("--save_dir", default=".", help="Directory to write flows_result*.json")
    args = ap.parse_args()

    prompts = load_yaml(args.prompts)
    if not isinstance(prompts, list):
        raise ValueError("prompts1.yaml must be a list of prompt objects with 'id' and 'text'.")

    gsm_items = [normalize_record(o) for o in read_jsonl_first_n(args.gsm_path, args.take_n)]
    fin_items = [normalize_record(o) for o in read_jsonl_first_n(args.finqa_path, args.take_n)]

    options = {
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    # Aggregation buckets by (dataset, prompt_id)
    buckets: Dict[Tuple[str,str], Dict[str, float]] = {}
    def ensure_bucket(ds: str, pid: str):
        if (ds, pid) not in buckets:
            buckets[(ds, pid)] = {
                "tot": 0,
                "em_ok": 0,
                "rnm_ok": 0,
                "sum_total_latency": 0.0,
            }

    # flows logging to file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    flows_path = Path(args.save_dir) / f"flows_result{ts}.json"
    flows_log: List[Dict[str, Any]] = []

    rows_print: List[Dict[str, Any]] = []  # for per-item summary

    for prompt_obj in prompts:
        pid = str(prompt_obj.get("id", "")).strip()
        ptext = str(prompt_obj.get("text", "")).strip()
        if not pid or not ptext:
            continue

        exec_kind = infer_exec_kind_from_id(pid)  # "program", "hybrid", or "text"

        for ds_name, items in [("GSM8K", gsm_items), ("FinQA", fin_items)]:
            for idx, ex in enumerate(items, 1):
                ensure_bucket(ds_name, pid)

                full_prompt = build_prompt_text(ptext, ex["question"], ex.get("context",""), args.sigfig, exec_kind)
                completion, llm_latency = ollama_generate(args.model, full_prompt, options)

                pred_text = extract_final_answer(completion)

                py_out = ""
                py_err = ""
                py_latency = 0.0
                code_used = ""

                if exec_kind in ("program", "hybrid"):
                    cand = select_python_block_for_question(completion, ex["question"])
                    if cand and is_safe_python(cand):
                        code_used = cand
                        py_out, py_err, py_latency, rc = run_python_code_sandboxed(cand, timeout_sec=6.0)
                        if rc == 0:
                            pot_pred = extract_answer_from_stdout(py_out)
                            if pot_pred:
                                pred_text = pot_pred

                total_latency = llm_latency + py_latency

                em_ok = exact_match_relaxed(pred_text, ex["gold"], rel_tol_em=0.0001)  # 0.01% relaxed EM
                rnm_ok = relative_numeric_match(pred_text, ex["gold"], args.rel_tol)

                b = buckets[(ds_name, pid)]
                b["tot"] += 1
                b["em_ok"] += int(bool(em_ok))
                b["rnm_ok"] += int(bool(rnm_ok))
                b["sum_total_latency"] += total_latency

                flow_entry = {
                    "dataset": ds_name,
                    "prompt_id": pid,
                    "item_index": idx,
                    "question": ex["question"],
                    "context": ex.get("context",""),
                    "gold": ex["gold"],
                    "prompt_text": full_prompt,
                    "completion": completion,
                    "pred": pred_text,
                    "em_match": bool(em_ok),
                    "rnm_match": bool(rnm_ok),
                    "llm_latency": llm_latency,
                    "py_latency": py_latency,
                    "total_latency": total_latency,
                }
                if exec_kind in ("program","hybrid"):
                    flow_entry.update({
                        "python_code": code_used,
                        "python_stdout": py_out,
                        "python_stderr": py_err,
                    })
                else:
                    flow_entry.update(parse_selfask_flow(completion))
                flows_log.append(flow_entry)

                # Per-item summary row
                rows_print.append({
                    "item_id": f"{ds_name}_{pid}_{idx}",
                    "dataset": ds_name,
                    "prompt_id": pid,
                    "pred": pred_text,
                    "gold": ex["gold"],
                    "em01pct": em_ok,     # EM with 0.01% relax
                    "rnm": rnm_ok,
                    "llm_latency": round(llm_latency, 3),
                    "py_latency": round(py_latency, 3),
                    "total_latency": round(total_latency, 3),
                })

    # Write flows file
    flows_path.parent.mkdir(parents=True, exist_ok=True)
    with open(flows_path, "w", encoding="utf-8") as f:
        json.dump(flows_log, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # Per-item summary printout
    # -----------------------------
    print("\n=== Per-item Summary (EM@0.01% relaxed, RNM@rel_tol) ===")
    colw = {
        "item_id": 30, "dataset": 8, "prompt_id": 28,
        "pred": 16, "gold": 14, "em": 8, "rnm": 8,
        "llm": 10, "py": 10, "tot": 10
    }
    header = (f'{"item_id":<{colw["item_id"]}}  '
              f'{"dataset":<{colw["dataset"]}}  '
              f'{"prompt_id":<{colw["prompt_id"]}}  '
              f'{"pred":<{colw["pred"]}}  '
              f'{"gold":<{colw["gold"]}}  '
              f'{"EM":<{colw["em"]}}  '
              f'{"RNM":<{colw["rnm"]}}  '
              f'{"llm_s":<{colw["llm"]}}  '
              f'{"py_s":<{colw["py"]}}  '
              f'{"total_s":<{colw["tot"]}}')
    print(header)
    print("-" * len(header))
    for r in rows_print:
        print(f'{r["item_id"]:<{colw["item_id"]}}  '
              f'{r["dataset"]:<{colw["dataset"]}}  '
              f'{r["prompt_id"]:<{colw["prompt_id"]}}  '
              f'{r["pred"]:<{colw["pred"]}}  '
              f'{r["gold"]:<{colw["gold"]}}  '
              f'{str(bool(r["em01pct"])):<{colw["em"]}}  '
              f'{str(bool(r["rnm"])):<{colw["rnm"]}}  '
              f'{r["llm_latency"]:<{colw["llm"]}}  '
              f'{r["py_latency"]:<{colw["py"]}}  '
              f'{r["total_latency"]:<{colw["tot"]}}')

    # -----------------------------
    # Final summary table
    # -----------------------------
    def pct(x, n):
        return f"{(100.0 * x / n):.1f}%" if n > 0 else "n/a"
    def avg_latency(sum_lat, n):
        return f"{(sum_lat / n):.2f}s" if n > 0 else "n/a"

    print("\n=== Accuracy & Avg Latency by Dataset × PromptID ===")
    wds, wpid, w1, w2, w3, wcnt = 8, 28, 8, 8, 18, 20
    header2 = (f'{"dataset":<{wds}}  {"prompt_id":<{wpid}}  '
               f'{"EM":<{w1}}  {"RNM":<{w2}}  {"avg_total_latency":<{w3}}  '
               f'{"(N EM / N total)":<{wcnt}}')
    print(header2)
    print("-" * len(header2))

    # stable ordering
    ordered_keys = sorted(buckets.keys(), key=lambda k: (k[0], k[1]))
    for ds, pid in ordered_keys:
        b = buckets[(ds, pid)]
        tot = int(b["tot"])
        em = pct(b["em_ok"], tot)
        rnm = pct(b["rnm_ok"], tot)
        lat = avg_latency(b["sum_total_latency"], tot)
        print(f'{ds:<{wds}}  {pid:<{wpid}}  {em:<{w1}}  {rnm:<{w2}}  {lat:<{w3}}  ({b["em_ok"]}/{tot})')

    # If you want to see where flows were written:
    # print(f"\nFlows written to: {flows_path}")

if __name__ == "__main__":
    main()
