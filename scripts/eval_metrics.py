import sys, pandas as pd, re, math
from pathlib import Path

def to_num(x):
    if x is None: return None
    s = str(x).strip()
    s = s.replace(",","")
    pct = None
    m = re.match(r"^\$?\s*([-+]?\d*\.?\d+)\s*%$", s)
    if m:
        pct = float(m.group(1)) / 100.0
        return ("pct", pct)
    m2 = re.match(r"^\$?\s*([-+]?\d*\.?\d+)$", s)
    if m2: return ("abs", float(m2.group(1)))
    return None

def main(csv_path):
    df = pd.read_csv(csv_path)
    # exact match (case-insensitive, strip)
    df["em"] = (df["gold"].fillna("").str.strip().str.lower()
                == df["pred"].fillna("").str.strip().str.lower())

    # numeric error when both parse as numbers
    errs = []
    for _, r in df.iterrows():
        g = to_num(r.get("gold"))
        p = to_num(r.get("pred"))
        if g and p and g[0] == p[0] == "abs" and g[1] != 0:
            errs.append(abs(p[1] - g[1]) / abs(g[1]))
        elif g and p and g[0] == p[0] == "pct" and g[1] != 0:
            errs.append(abs(p[1] - g[1]) / abs(g[1]))
        else:
            errs.append(math.nan)
    df["rel_numeric_err"] = errs

    # cross-domain slices
    def dom_mask(dataset_name):
        return df["dataset"].str.contains(dataset_name, case=False, na=False)

    report = {
        "N_total": len(df),
        "EM_overall": float(df["em"].mean()),
        "Avg_rel_numeric_err": float(df["rel_numeric_err"].mean(skipna=True)),
        "EM_finqa": float(df[dom_mask("finqa")]["em"].mean()),
        "EM_convfinqa": float(df[dom_mask("convfinqa")]["em"].mean()),
        "EM_tatqa": float(df[dom_mask("tatqa")]["em"].mean()),
        "EM_gsm8k": float(df[dom_mask("gsm8k")]["em"].mean()),
    }
    print(report)

if __name__ == "__main__":
    main(sys.argv[1])
