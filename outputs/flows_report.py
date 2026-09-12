# flows_metrics_report.py
# Usage:
#   python flows_metrics_report.py /path/to/flows_resultYYYYMMDD_HHMMSS.json

import sys, json, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
assert in_path and in_path.exists(), f"Provide a valid path to flows_result*.json (got: {in_path})"

with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# --- helpers ---
NUM_RE = re.compile(r"""(?xi)
^\s*\$?\s*(?P<sign>[-+])?\s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)
\s*(?P<pct>%?)\s*$
""")

def parse_num(s):
    if s is None: return None
    s = str(s).strip()
    m = NUM_RE.match(s)
    if not m: return None
    num = float((m.group("sign") or "") + m.group("num").replace(",",""))
    typ = "pct" if m.group("pct") else "abs"
    return typ, num

def rel_numeric_match(pred, gold, tol=0.01):
    p, g = parse_num(pred), parse_num(gold)
    if not p or not g or p[0] != g[0]:
        return False
    if g[1] == 0:
        return abs(p[1] - g[1]) <= tol
    return abs(p[1]-g[1]) / abs(g[1]) <= tol

def pct_abs_mismatch(pred, gold):
    p, g = parse_num(pred), parse_num(gold)
    return bool(p and g and p[0] != g[0])

def categorize_error(row):
    # correct -> none
    if row.get("em_match") or row.get("rnm_match"): return "none"
    # execution errors for PoT/Hybrid when stderr present
    pid = str(row.get("prompt_id") or "").lower()
    uses_python = ("program" in pid) or ("hybrid" in pid) or ("pot" in pid)
    py_err = (row.get("py_stderr") or "").strip()
    if uses_python and py_err:
        return "execution"
    # formatting if cannot parse numeric pred
    if parse_num(row.get("pred")) is None:
        return "formatting"
    # extraction if pct/abs mismatch
    if pct_abs_mismatch(row.get("pred"), row.get("gold")):
        return "extraction"
    # arithmetic if both numeric types match but value off
    if parse_num(row.get("pred")) and parse_num(row.get("gold")):
        return "arithmetic"
    return "extraction"

def first_error_position(pred, gold):
    p, g = parse_num(pred), parse_num(gold)
    if not p or not g: return np.nan
    def norm(v):
        s = f"{'' if v>=0 else '-'}{abs(v)}"
        if "." in s: s = s.rstrip("0").rstrip(".")
        return s
    ps, gs = norm(p[1]), norm(g[1])
    L = min(len(ps), len(gs))
    for i in range(L):
        if ps[i] != gs[i]: return i
    return L if len(ps) != len(gs) else np.nan

# --- derived columns ---
df["rel_1pct"] = [rel_numeric_match(r.get("pred"), r.get("gold"), 0.01) for _, r in df.iterrows()]
df["err_category"] = [categorize_error(r) for _, r in df.iterrows()]
df["first_error_pos"] = [first_error_position(r.get("pred"), r.get("gold")) for _, r in df.iterrows()]

pid_lower = df["prompt_id"].astype(str).str.lower()
df["used_python"] = pid_lower.str.contains("program|hybrid|pot", regex=True)
df["syntax_error"] = (df.get("py_stderr", "") \
                      .fillna("") \
                      .astype(str) \
                      .str.lower() \
                      .str.contains("syntaxerror"))

# Incorrect Python code but correct answer
no_code_msg = df.get("py_stderr","").fillna("").astype(str).str.lower().str.contains("no safe python block found")
df["inc_code_but_correct"] = (df["used_python"] & (df["em_match"] | df["rnm_match"]) & (df["syntax_error"] | no_code_msg | (df.get("py_stderr","").fillna("")!="")))

# --- per-item summary (like your original), with tighter EM that allows 0.01% numeric tolerance ---
def tol_em(pred, gold, tol=0.0001):  # 0.01% for “exact” to avoid sig-fig false negatives
    # If both parse as numbers and same unit type, use 0.01% tolerance. Else fall back to string-equal.
    p, g = parse_num(pred), parse_num(gold)
    if p and g and p[0]==g[0] and g[1]!=0:
        return abs(p[1]-g[1]) / abs(g[1]) <= tol
    return str(pred).strip().lower() == str(gold).strip().lower()

df["em_tight_0p01pct"] = [tol_em(r.get("pred"), r.get("gold"), 0.0001) for _, r in df.iterrows()]

# Save per-item summary CSV
per_item_csv = in_path.with_name("flows_annotated_metrics.csv")
df.to_csv(per_item_csv, index=False)

# --- aggregate summaries by prompt_id ---
grp = df.groupby("prompt_id", dropna=False)

err_df = grp.apply(lambda g: pd.Series({
    "N": int(len(g)),
    "EM_rate_tight0.01pct": float(g["em_tight_0p01pct"].mean()),
    "RNM_rate_1pct": float(g["rel_1pct"].mean()),
    "total_wrong": int((~(g["em_tight_0p01pct"] | g["rel_1pct"])).sum()),
    "extraction": int((g["err_category"]=="extraction").sum()),
    "arithmetic": int((g["err_category"]=="arithmetic").sum()),
    "formatting": int((g["err_category"]=="formatting").sum()),
    "execution": int((g["err_category"]=="execution").sum()),
    "mean_first_error_pos": float(pd.to_numeric(g.loc[~(g["em_tight_0p01pct"] | g["rel_1pct"]),"first_error_pos"]).mean()),
    "median_first_error_pos": float(pd.to_numeric(g.loc[~(g["em_tight_0p01pct"] | g["rel_1pct"]),"first_error_pos"]).median()),
    "incorrect_code_but_correct": int(g["inc_code_but_correct"].sum()),
    "syntax_error_count": int(g.loc[g["used_python"], "syntax_error"].sum()),
    "used_python_n": int(g["used_python"].sum()),
    "avg_llm_latency_s": float(g["llm_latency"].mean()),
    "avg_py_latency_s": float(g["py_latency"].mean()),
    "avg_total_latency_s": float(g["total_latency"].mean())
})).reset_index()

for cat in ["extraction","arithmetic","formatting","execution"]:
    err_df[f"{cat}_share"] = err_df.apply(lambda r: r[cat]/r["total_wrong"] if r["total_wrong"]>0 else 0.0, axis=1)

summary_csv = in_path.with_name("flows_metrics_summary_by_prompt.csv")
err_df.to_csv(summary_csv, index=False)

# --- stacked bar: error attribution shares per prompt_id ---
plt.figure(figsize=(10,6))
e = err_df.sort_values("total_wrong", ascending=False)
x = np.arange(len(e))
bottom = np.zeros(len(e))
for cat, label in zip(["extraction","arithmetic","formatting","execution"],
                      ["Extraction","Arithmetic","Formatting","Execution"]):
    vals = e[f"{cat}_share"].values
    plt.bar(x, vals, width=0.6, bottom=bottom, label=label)
    bottom += vals
plt.xticks(x, e["prompt_id"].astype(str).tolist(), rotation=25, ha="right")
plt.ylim(0,1)
plt.ylabel("Share of errors")
plt.title("Error attribution by prompt")
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend(loc="upper right", frameon=False)
plot_path = in_path.with_name("error_attribution_stacked_bar.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=200)

# --- print a concise report ---
print("\n== Aggregate metrics by prompt_id ==")
print(err_df[[
    "prompt_id","N","EM_rate_tight0.01pct","RNM_rate_1pct",
    "avg_llm_latency_s","avg_py_latency_s","avg_total_latency_s"
]].to_string(index=False))

print("\n== Error attribution counts and shares ==")
print(err_df[[
    "prompt_id","total_wrong","extraction","arithmetic","formatting","execution",
    "extraction_share","arithmetic_share","formatting_share","execution_share"
]].to_string(index=False))

print("\n== First-error position (incorrect only) ==")
print(err_df[["prompt_id","mean_first_error_pos","median_first_error_pos"]].to_string(index=False))

print("\n== Python-specific diagnostics ==")
print(err_df[[
    "prompt_id","used_python_n","syntax_error_count","incorrect_code_but_correct"
]].to_string(index=False))

print(f"\nSaved per-item CSV: {per_item_csv}")
print(f"Saved summary CSV:  {summary_csv}")
print(f"Saved plot PNG:     {plot_path}")
