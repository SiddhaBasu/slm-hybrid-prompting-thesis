# scripts/unify_datasets.py
import json, re
from pathlib import Path
from tqdm import tqdm

RAW = Path("data/raw")
UNI = Path("data/unified")
UNI.mkdir(parents=True, exist_ok=True)

def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def unify_finqa(in_path, out_path):
    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex.get("question") or ex.get("Query") or ""

            # want 'answer'; fall back to 'final_result' if blank
            ans = (ex.get("answer") or "").strip()
            if not ans:
                fr = ex.get("final_result")
                ans = str(fr).strip() if fr is not None else ""

            # pre/post can be lists; join them
            pre = ex.get("pre_text", [])
            post = ex.get("post_text", [])
            if isinstance(pre, list): pre = " ".join(pre)
            if isinstance(post, list): post = " ".join(post)

            # Table -> simple linearization (first row as header if present)
            tbl = ex.get("table", [])
            tbl_str = ""
            if isinstance(tbl, list) and tbl and isinstance(tbl[0], list):
                hdr = tbl[0]
                for r in tbl[1:6]:  # cap to avoid long contexts
                    row = dict(zip(hdr, r))
                    tbl_str += " | ".join(f"{k}={v}" for k, v in row.items()) + "\n"

            context = (pre + ("\n" if pre else "") + tbl_str + (("\n" + post) if post else "")).strip()

            rows.append({
                "id": ex.get("id", "finqa_unk"),
                "domain": "finance",
                "dataset": "finqa",
                "question": q,
                "context": context,
                "numbers_json": [],
                "answer": ans,
            })
    write_jsonl(rows, out_path)

def unify_convfinqa(in_path, out_path):
    rows = []
    for line in open(in_path, "r", encoding="utf-8"):
        ex = json.loads(line)
        # mirrors vary; we expect fields like conversation or turns
        q = ex.get("question") or ex.get("final_question") or ""
        ans = ex.get("answer") or ex.get("final_answer") or ""
        ctx = ex.get("context") or ""
        rows.append({
            "id": ex.get("id","convfinqa_unk"),
            "domain":"finance",
            "dataset":"convfinqa",
            "question": q,
            "context": ctx,
            "numbers_json": [],
            "answer": ans
        })
    write_jsonl(rows, out_path)

def unify_tatqa(in_path, out_path):
    rows = []
    for line in open(in_path, "r", encoding="utf-8"):
        ex = json.loads(line)
        q = ex.get("question","")
        ans = ex.get("answer","")
        # table/text fields differ; fall back gracefully
        ctx_parts = []
        for k in ("table","paragraphs","pre_text","post_text","content"):
            if k in ex:
                ctx_parts.append(str(ex[k])[:4000])  # cap
        context = "\n".join(ctx_parts)
        rows.append({
            "id": ex.get("id","tatqa_unk"),
            "domain":"finance",
            "dataset":"tatqa",
            "question": q,
            "context": context,
            "numbers_json": [],
            "answer": ans
        })
    write_jsonl(rows, out_path)

def unify_gsm8k(in_path, out_path):
    # HF gsm8k-platinum has fields: question, answer
    rows = []
    for line in open(in_path, "r", encoding="utf-8"):
        ex = json.loads(line)
        q = ex.get("question","")
        a = ex.get("answer","")
        # Extract final numeric answer if present (common GSM8K format)
        m = re.search(r"####\s*([^\n]+)", a)
        ans = m.group(1).strip() if m else a.strip()
        rows.append({
            "id": ex.get("id", ex.get("question_id","gsm8k_unk")),
            "domain":"math",
            "dataset":"gsm8k",
            "question": q,
            "context": "",
            "numbers_json": [],
            "answer": ans
        })
    write_jsonl(rows, out_path)

if __name__ == "__main__":
    unify_finqa("data/raw/finqa_val_100.jsonl", "data/unified/finqa_100.jsonl")
    #unify_convfinqa("data/raw/convfinqa_200.jsonl", "data/unified/convfinqa_200.jsonl")
    #unify_tatqa("data/raw/tatqa_200.jsonl", "data/unified/tatqa_200.jsonl")
    unify_gsm8k("data/raw/gsm8k_platinum_200.jsonl", "data/unified/gsm8k_200.jsonl")
    print("Unified files in data/unified/*")
