# scripts/seek_prompts.py
# Quick sanity runner for Self-Ask, Direct, Program (PoT), and Hybrid (Self-Ask + PoT) prompts
# across GSM8K and FinQA. Revised to select the correct python block per item/question.

import argparse
import json
import time
import yaml
import requests
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# -----------------------------
# Regex Helpers
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

FENCED_PY_RE = re.compile(
    r"```(?:python)?\s*(?P<code>[\s\S]*?)\s*```",
    flags=re.IGNORECASE
)

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

def tol_equal(pred: str, gold: str, rel_tol: float = 0.01) -> bool:
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
# Self-Ask Flow Parsing (debug)
# -----------------------------

def parse_selfask_flow(text: str) -> Dict[str, List[str]]:
    questions: List[str] = []
    answers: List[str] = []
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        mq = FLOW_Q_RE.match(line_stripped)
        if mq:
            q = mq.group(1).strip()
            if q and q not in questions:
                questions.append(q)
            continue
        ma = FLOW_A_RE.match(line_stripped)
        if ma:
            a = ma.group(1).strip()
            if a and a not in answers:
                answers.append(a)
            continue
        if line_stripped.endswith("?") and len(line_stripped) <= 180:
            if line_stripped not in questions:
                questions.append(line_stripped)
    return {"questions": questions, "answers": answers}

# -----------------------------
# Program-of-Thought (PoT) Helpers
# -----------------------------

UNSAFE_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+sys\b",
    r"\bopen\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\brequests\b",
    r"\burllib\b",
    r"\bsocket\b",
    r"\bhttp\b",
    r"\bPopen\b",
]

SAFE_IMPORT_ALLOWLIST = [
    "math",
    "statistics",
    "re",
    "fractions",
    "decimal",
]

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

NUM_TOKEN_RE = re.compile(r"(?<![A-Za-z_])(-?\d+(?:\.\d+)?)")

def numbers_in_text(s: str) -> List[str]:
    # capture bare numbers; currency or % stripped earlier by normalizers
    return [m.group(1) for m in NUM_TOKEN_RE.finditer(s or "")]

def score_code_block(code: str, q_numbers: List[str]) -> float:
    score = 0.0
    # prefer a print of final value
    if re.search(r"\bprint\s*\(", code):
        score += 2.0
    # overlap with numbers from the question
    # compare as floats with a small tolerance
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
    # penalize suspicious placeholders
    if re.search(r"\bTODO\b|\bpass\b|NotImplementedError", code):
        score -= 1.0
    return score

def select_python_block_for_question(completion: str, question: str) -> Optional[str]:
    blocks = extract_all_python_blocks(completion)
    if not blocks:
        return None
    q_nums = numbers_in_text(question)
    # score each, prefer later tie-break
    scored = []
    for idx, (start, end, code) in enumerate(blocks):
        s = score_code_block(code, q_nums)
        scored.append((s, start, code))
    scored.sort(key=lambda t: (t[0], t[1]))  # by score then position
    best = scored[-1]  # highest score, latest if tie
    return best[2] if best[0] > 0.0 else blocks[-1][2]  # if all zero, fall back to last

def run_python_code_sandboxed(code: str, timeout_sec: float = 5.0) -> Tuple[str, str, float, int]:
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
                    help="Path to GSM8K jsonl (first N items used).")
    ap.add_argument("--finqa_path", default="data/unified/finqa_100.jsonl",
                    help="Path to FinQA jsonl (first N items used).")
    ap.add_argument("--take_n", type=int, default=5,
                    help="Number of items to take from each dataset.")
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--num_predict", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sigfig", type=int, default=3,
                    help="If numeric, encourage models to round/print to this many significant figures.")
    ap.add_argument("--show_flow", action="store_true",
                    help="If set, prints FULL completion and parsed flow (Self-Ask only).")
    ap.add_argument("--enable_pot", action="store_true",
                    help="If set, also runs program(PoT) prompts and executes code.")
    ap.add_argument("--hybrid_only", action="store_true",
                    help="Run ONLY the hybrid prompts. Implies --enable_pot.")
    ap.add_argument("--save_json", type=str, default="",
                    help="Optional path to save raw outputs as JSON (for debugging).")
    args = ap.parse_args()

    prompts = load_yaml(args.prompts)

    def find_prompt(domain_name: str, type_name: str) -> Optional[Dict[str,Any]]:
        return next((p for p in prompts
                     if str(p.get("domain","")).lower()==domain_name
                     and str(p.get("type","")).lower()==type_name
                     and "text" in p), None)

    def find_prompt_by_id(pid: str) -> Optional[Dict[str,Any]]:
        return next((p for p in prompts if str(p.get("id","")).lower()==pid.lower()), None)

    # Prefer true hybrids if present
    math_hybrid = find_prompt_by_id("math_hybrid_selfask_pot_v1")
    fin_hybrid  = find_prompt_by_id("finance_hybrid_selfask_pot_v1") or find_prompt_by_id("finance_hybrid_selfask_pot")

    # Select prompts
    math_selfask = math_direct = math_program = None
    fin_selfask  = fin_direct  = fin_program  = None

    if args.hybrid_only:
        args.enable_pot = True
        math_program = math_hybrid
        fin_program  = fin_hybrid
        if not any([math_program, fin_program]):
            print("WARNING: --hybrid_only set but no hybrid prompts found in prompts.yaml.")
    else:
        math_selfask = find_prompt("math", "selfask")
        math_direct  = find_prompt("math", "direct")
        math_program = find_prompt("math", "program") if args.enable_pot else None
        fin_selfask  = find_prompt("finance", "selfask")
        fin_direct   = find_prompt("finance", "direct")
        fin_program  = find_prompt("finance", "program") if args.enable_pot else None

        # If hybrids exist but plain program missing, use hybrids as program (still tracked as "hybrid" later)
        if args.enable_pot and not math_program and math_hybrid:
            math_program = math_hybrid
        if args.enable_pot and not fin_program and fin_hybrid:
            fin_program = fin_hybrid

    gsm_items   = read_jsonl_first_n(args.gsm_path, args.take_n) if any([math_selfask, math_direct, math_program]) else []
    finqa_items = read_jsonl_first_n(args.finqa_path, args.take_n) if any([fin_selfask, fin_direct, fin_program]) else []

    gsm_norm = [normalize_record(o, "math") for o in gsm_items]
    fin_norm = [normalize_record(o, "finance") for o in finqa_items]

    options = {
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": args.temperature,
        "seed": args.seed,
    }

    print("\n=== Running prompts (Self-Ask / Direct / Program[PoT] / Hybrid) on GSM8K + FinQA ===")
    print(f"Model: {args.model}")
    print(f"Decoding: num_ctx={args.num_ctx}, num_predict={args.num_predict}, "
          f"temperature={args.temperature}, seed={args.seed}, sigfig={args.sigfig}")
    if args.hybrid_only:
        print("Mode: HYBRID ONLY\n")
    else:
        print(f"enable_pot={args.enable_pot}\n")

    rows: List[Dict[str, Any]] = []
    raw_log: List[Dict[str, Any]] = []

    buckets: Dict[Tuple[str,str], Dict[str, float]] = {
        ("gsm8k", "selfask"): {"ok": 0, "tot": 0, "sum_llm": 0.0, "sum_py": 0.0, "sum_total": 0.0},
        ("gsm8k", "direct"):  {"ok": 0, "tot": 0, "sum_llm": 0.0, "sum_py": 0.0, "sum_total": 0.0},
        ("gsm8k", "program"): {"ok": 0, "tot": 0, "sum_llm": 0.0, "sum_py": 0.0, "sum_total": 0.0},
        ("gsm8k", "hybrid"):  {"ok": 0, "tot": 0, "sum_llm": 0.0, "sum_py": 0.0, "sum_total": 0.0},
        ("finqa", "selfask"): {"ok": 0, "tot": 0, "sum_llm": 0.0, "sum_py": 0.0, "sum_total": 0.0},
        ("finqa", "direct"):  {"ok": 0, "tot": 0, "sum_llm": 0.0, "sum_py": 0.0, "sum_total": 0.0},
        ("finqa", "program"): {"ok": 0, "tot": 0, "sum_llm": 0.0, "sum_py": 0.0, "sum_total": 0.0},
        ("finqa", "hybrid"):  {"ok": 0, "tot": 0, "sum_llm": 0.0, "sum_py": 0.0, "sum_total": 0.0},
    }

    def style_label_for(prompt_obj: Dict[str,Any]) -> str:
        ptype = str(prompt_obj.get("type","")).lower()
        pid   = str(prompt_obj.get("id","")).lower()
        if "hybrid_selfask_pot" in pid:
            return "hybrid"
        return ptype

    def build_prompt_text(prompt_obj: Dict[str,Any], q: str, ctx: str) -> str:
        base = (prompt_obj["text"]
                    .replace("{QUESTION}", q)
                    .replace("{CONTEXT}", ctx or "(none)")
                    .replace("{NUMBERS_JSON}", "[]")
                    .replace("{SIGFIG}", str(args.sigfig)))
        ptype = str(prompt_obj.get("type","")).lower()
        pid   = str(prompt_obj.get("id","")).lower()
        is_hybrid = ("hybrid_selfask_pot" in pid)
        if ptype == "program" or is_hybrid:
            extra = f"\n\n[Formatting note] In your Python, format/round the final printed numeric value to {args.sigfig} significant figures if applicable."
            return base + extra
        else:
            extra = (
                f"\n\n[Formatting]\n"
                f"- If the answer is numeric, round to {args.sigfig} significant figures.\n"
                f"- End with a single line: Answer: <value>\n"
            )
            return base + extra

    def run_one(example: Dict[str,Any], prompt_obj: Dict[str,Any], domain_tag: str) -> Dict[str,Any]:
        full_prompt = build_prompt_text(prompt_obj, example["question"], example.get("context",""))
        completion, llm_latency = ollama_generate(args.model, full_prompt, options)

        ptype = str(prompt_obj.get("type","")).lower()
        pid   = str(prompt_obj.get("id","")).lower()
        is_hybrid = ("hybrid_selfask_pot" in pid)
        style = "hybrid" if is_hybrid else ptype

        pred = extract_final_answer(completion)
        py_out = ""
        py_err = ""
        py_rt  = 0.0
        code_used = ""

        if style in ("program", "hybrid"):
            # NEW: choose the python block that best matches THIS question.
            cand = select_python_block_for_question(completion, example["question"])
            if cand and is_safe_python(cand):
                code_used = cand
                py_out, py_err, py_rt, rc = run_python_code_sandboxed(cand, timeout_sec=6.0)
                if rc == 0:
                    pot_pred = extract_answer_from_stdout(py_out)
                    if pot_pred:
                        pred = pot_pred
                # else: keep text pred
            else:
                py_err = "No safe or relevant python block found."

        total_latency = llm_latency + py_rt
        ok = tol_equal(pred, example["gold"], rel_tol=0.01)

        if (domain_tag, style) in buckets:
            buckets[(domain_tag, style)]["tot"] += 1
            buckets[(domain_tag, style)]["ok"]  += int(bool(ok))
            buckets[(domain_tag, style)]["sum_llm"]   += llm_latency
            buckets[(domain_tag, style)]["sum_py"]    += py_rt
            buckets[(domain_tag, style)]["sum_total"] += total_latency

        return {
            "prompt_text": full_prompt,
            "completion": completion,
            "ptype": ptype,
            "style": style,
            "pred": pred,
            "ok": ok,
            "llm_latency": llm_latency,
            "py_latency": py_rt,
            "latency_sec": total_latency,
            "py_stdout": py_out,
            "py_stderr": py_err,
            "py_code": code_used,
        }

    def run_batch(examples: List[Dict[str,Any]], prompt_obj: Dict[str,Any], domain_tag: str, label_prefix: str):
        for idx, ex in enumerate(examples, 1):
            res = run_one(ex, prompt_obj, domain_tag)
            rid = f"{domain_tag}_{label_prefix}_{idx}"
            rows.append({
                "item_id": rid,
                "domain": domain_tag,
                "prompt_id": prompt_obj.get("id", f"{domain_tag}_{label_prefix}"),
                "ptype": res["ptype"],
                "style": res["style"],
                "pred": res["pred"],
                "gold": ex["gold"],
                "correct_tol1pct": bool(res["ok"]),
                "llm_latency": round(res["llm_latency"], 3),
                "py_latency": round(res["py_latency"], 3),
                "latency_sec": round(res["latency_sec"], 3),
            })
            raw_log.append({
                "item_id": rid,
                "domain": domain_tag,
                "prompt_id": prompt_obj.get("id", f"{domain_tag}_{label_prefix}"),
                "ptype": res["ptype"],
                "style": res["style"],
                "prompt_text": res["prompt_text"],
                "completion": res["completion"],
                "pred": res["pred"],
                "gold": ex["gold"],
                "llm_latency": res["llm_latency"],
                "py_latency": res["py_latency"],
                "latency_sec": res["latency_sec"],
                "py_stdout": res["py_stdout"],
                "py_stderr": res["py_stderr"],
                "py_code": res["py_code"],
            })

            if args.show_flow and res["style"] == "selfask":
                print("=" * 88)
                print(f"[{rid}] Prompt: {prompt_obj.get('id', f'{domain_tag}_selfask')}")
                print("-" * 88)
                print(">> FULL COMPLETION:")
                print(res["completion"].strip())
                flow = parse_selfask_flow(res["completion"])
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
                print(f"Extracted pred = {res['pred']}   |   GOLD = {ex['gold']}   |   ok@1% = {res['ok']}")
                print(f"LLM latency = {res['llm_latency']:.2f}s  |  Py latency = {res['py_latency']:.2f}s  |  Total = {res['latency_sec']:.2f}s")
                print("=" * 88 + "\n")

            if args.show_flow and res["style"] in ("program","hybrid"):
                print("=" * 88)
                print(f"[{rid}] Prompt: {prompt_obj.get('id', f'{domain_tag}_{res['style']}')}")
                print("-" * 88)
                print(">> FULL COMPLETION:")
                print(res["completion"].strip())
                print("\n>> CHOSEN PYTHON CODE:")
                print(res["py_code"] or "(none)")
                print("\n>> PYTHON STDOUT:")
                print((res["py_stdout"] or "").strip() or "(empty)")
                if res["py_stderr"].strip():
                    print("\n>> PYTHON STDERR:")
                    print(res["py_stderr"].strip())
                print("-" * 88)
                print(f"Extracted pred = {res['pred']}   |   GOLD = {ex['gold']}   |   ok@1% = {res['ok']}")
                print(f"LLM latency = {res['llm_latency']:.2f}s  |  Py latency = {res['py_latency']:.2f}s  |  Total = {res['latency_sec']:.2f}s")
                print("=" * 88 + "\n")

    # ---- Run GSM8K ----
    if gsm_norm:
        if args.hybrid_only:
            if math_hybrid:
                print(f"Running GSM8K (first {len(gsm_norm)}) with HYBRID PoT: {math_hybrid.get('id')}")
                run_batch(gsm_norm, math_hybrid, "gsm8k", "hybrid")
        else:
            if math_hybrid and args.enable_pot:
                print(f"Running GSM8K with HYBRID PoT: {math_hybrid.get('id')}")
                run_batch(gsm_norm, math_hybrid, "gsm8k", "hybrid")
            if math_selfask:
                print(f"Running GSM8K with selfask: {math_selfask.get('id','math_selfask')}")
                run_batch(gsm_norm, math_selfask, "gsm8k", "selfask")
            if math_direct:
                print(f"Running GSM8K with direct: {math_direct.get('id','math_direct')}")
                run_batch(gsm_norm, math_direct, "gsm8k", "direct")
            if args.enable_pot:
                if math_program and (not math_hybrid or math_program is not math_hybrid):
                    print(f"Running GSM8K with program (PoT): {math_program.get('id','math_program')}")
                    run_batch(gsm_norm, math_program, "gsm8k", "program")

    # ---- Run FinQA ----
    if fin_norm:
        if args.hybrid_only:
            if fin_hybrid:
                print(f"\nRunning FinQA (first {len(fin_norm)}) with HYBRID PoT: {fin_hybrid.get('id')}")
                run_batch(fin_norm, fin_hybrid, "finqa", "hybrid")
        else:
            if fin_hybrid and args.enable_pot:
                print(f"\nRunning FinQA with HYBRID PoT: {fin_hybrid.get('id')}")
                run_batch(fin_norm, fin_hybrid, "finqa", "hybrid")
            if fin_selfask:
                print(f"Running FinQA with selfask: {fin_selfask.get('id','finance_selfask')}")
                run_batch(fin_norm, fin_selfask, "finqa", "selfask")
            if fin_direct:
                print(f"Running FinQA with direct: {fin_direct.get('id','finance_direct')}")
                run_batch(fin_norm, fin_direct, "finqa", "direct")
            if args.enable_pot:
                if fin_program and (not fin_hybrid or fin_program is not fin_hybrid):
                    print(f"Running FinQA with program (PoT): {fin_program.get('id','finance_program')}")
                    run_batch(fin_norm, fin_program, "finqa", "program")

    # ---- Per-item summary table ----
    print("\n=== Per-item Summary (ok@1% tolerance) ===")
    colw = {"item_id": 22, "domain": 8, "style": 8, "prompt_id": 32, "pred": 16, "gold": 14, "ok": 8, "llm": 10, "py": 10, "tot": 10}
    header = (f'{"item_id":<{colw["item_id"]}}  '
              f'{"domain":<{colw["domain"]}}  '
              f'{"style":<{colw["style"]}}  '
              f'{"prompt_id":<{colw["prompt_id"]}}  '
              f'{"pred":<{colw["pred"]}}  '
              f'{"gold":<{colw["gold"]}}  '
              f'{"ok@1%":<{colw["ok"]}}  '
              f'{"llm_s":<{colw["llm"]}}  '
              f'{"py_s":<{colw["py"]}}  '
              f'{"total_s":<{colw["tot"]}}')
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f'{r["item_id"]:<{colw["item_id"]}}  '
              f'{r["domain"]:<{colw["domain"]}}  '
              f'{r["style"]:<{colw["style"]}}  '
              f'{r["prompt_id"]:<{colw["prompt_id"]}}  '
              f'{r["pred"]:<{colw["pred"]}}  '
              f'{r["gold"]:<{colw["gold"]}}  '
              f'{str(r["correct_tol1pct"]):<{colw["ok"]}}  '
              f'{r["llm_latency"]:<{colw["llm"]}}  '
              f'{r["py_latency"]:<{colw["py"]}}  '
              f'{r["latency_sec"]:<{colw["tot"]}}')

    # ---- Accuracy + Average Latency table ----
    def pct(ok, tot):
        return f"{(100.0 * ok / tot):.1f}%" if tot > 0 else "n/a"

    def avg(val_sum, tot):
        return f"{(val_sum / tot):.2f}s" if tot > 0 else "n/a"

    print("\n=== Accuracy & Avg Latency by Dataset × Style ===")
    acc_rows = [
        ("GSM8K", "selfask"),
        ("GSM8K", "direct"),
        ("GSM8K", "program"),
        ("GSM8K", "hybrid"),
        ("FinQA", "selfask"),
        ("FinQA", "direct"),
        ("FinQA", "program"),
        ("FinQA", "hybrid"),
    ]
    wds, wst, wacc, wlat = 8, 8, 10, 12
    print(f'{"dataset":<{wds}}  {"style":<{wst}}  {"accuracy":<{wacc}}  {"avg_total_latency":<{wlat}}  (N correct / N total)')
    print("-" * (wds + wst + wacc + wlat + 18))
    for ds, st in acc_rows:
        key = ("gsm8k" if ds.lower()=="gsm8k" else "finqa", st)
        ok = int(buckets[key]["ok"])
        tot = int(buckets[key]["tot"])
        acc = pct(ok, tot)
        avg_tot = avg(buckets[key]["sum_total"], tot)
        print(f'{ds:<{wds}}  {st:<{wst}}  {acc:<{wacc}}  {avg_tot:<{wlat}}  ({ok}/{tot})')

    if args.save_json:
        outp = Path(args.save_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "decoding": {
                    "num_ctx": args.num_ctx,
                    "num_predict": args.num_predict,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "sigfig": args.sigfig,
                },
                "runs": rows,
                "buckets": buckets,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved raw outputs to: {outp}")

    print("\nDone. To reduce mismatches further, keep PoT prompts zero-shot and avoid embedding few-shot code blocks in the model output.\n")

if __name__ == "__main__":
    main()
