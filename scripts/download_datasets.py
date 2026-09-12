# scripts/download_datasets.py
import os, json
from datasets import load_dataset
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def grab_finqa():
    # FinQA uses a custom loading script; enable remote code
    ds = load_dataset("ibm-research/finqa", trust_remote_code=True)  # splits: train/validation/test
    # print(list(ds.keys()))  # -> ['train', 'validation', 'test']
    save_jsonl(ds["validation"].select(range(100)), RAW/"finqa_val_100.jsonl")


def grab_convfinqa():
    # Several mirrors exist; choose one stable small mirror
    ds = load_dataset("ravithejads/convfinqa")  # turn-level format (≈3.2k rows)
    save_jsonl(ds["train"].select(range(200)), RAW/"convfinqa_200.jsonl")

def grab_tatqa():
    ds = load_dataset("next-tat/TAT-QA")
    save_jsonl(ds["train"].select(range(200)), RAW/"tatqa_200.jsonl")

def note_docfinqa():
    # DocFinQA is long-context; the ACL page hosts the paper/data pointers.
    # We'll just write a README note for you to fetch larger subsets later.
    (RAW/"README_DocFinQA.txt").write_text(
        "See DocFinQA ACL/ArXiv pages for data and instructions.\n", encoding="utf-8"
    )

def grab_gsm8k():
    # Use the standard HF mirror or a curated test split (e.g., madrylab/platinum)
    ds = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
    # take first 200 for quick runs
    save_jsonl(ds.select(range(200)), RAW/"gsm8k_platinum_200.jsonl")

def grab_math():
    # MATH isn’t on HF as a single loader; many use the GitHub repo.
    # We'll just write a note and you can clone later if needed.
    (RAW/"README_MATH.txt").write_text(
        "Get MATH from https://github.com/hendrycks/math and choose subsets.\n", encoding="utf-8"
    )

if __name__ == "__main__":
    grab_finqa()
    #grab_convfinqa()
    #grab_tatqa()
    #note_docfinqa()
    grab_gsm8k()
    #grab_math()
    print("Downloaded small slices to data/raw/*")
