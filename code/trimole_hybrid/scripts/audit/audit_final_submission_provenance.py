from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
FINAL_CSV = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission.csv"
OUT_CSV = ROOT / "results/model_log/final_validation_selected_submission/final_submission_provenance_audit.csv"

RUN_META_NAMES = {
    "00_run_meta.txt",
    "run_args.json",
    "run_args_fixed.json",
    "task_summary.json",
}

LOG_PATTERNS = ["*.log", "*.jsonl", "*.txt", "*.json"]

def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def classify_text(txt: str):
    hits = []
    if re.search(r"data/data_benchmark|data_benchmark", txt):
        hits.append("data_benchmark")
    if re.search(r"data/data_new|data_new", txt):
        hits.append("data_new")
    hits = list(dict.fromkeys(hits))
    if "data_benchmark" in hits and "data_new" in hits:
        return "mixed", hits
    if "data_benchmark" in hits:
        return "benchmark", hits
    if "data_new" in hits:
        return "data_new", hits
    return "unknown", hits

def existing_path_from_cell(cell):
    out = []
    if pd.isna(cell):
        return out
    s = str(cell).strip()
    if not s:
        return out
    for part in [x.strip() for x in s.split(" + ")]:
        if "::" in part:
            continue
        p = ROOT / part if not part.startswith("/") else Path(part)
        if p.exists():
            out.append(p)
    return out

def candidate_files_from_path(p: Path):
    cands = []
    if p.is_file():
        cands.append(p)
        parent = p.parent
    else:
        parent = p

    dirs = [parent]
    cur = parent
    for _ in range(3):
        cur = cur.parent
        dirs.append(cur)

    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        for name in RUN_META_NAMES:
            f = d / name
            if f.exists() and f.is_file():
                cands.append(f)
        for pat in LOG_PATTERNS:
            cands.extend(sorted(d.glob(pat)))

    uniq = []
    seen = set()
    for x in cands:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            uniq.append(x)
    return uniq

df = pd.read_csv(FINAL_CSV)
rows = []

for _, row in df.iterrows():
    task = str(row["task"])
    seed_sources = []

    for c in ["rerun_results_file", "rerun_task_dir"]:
        if c in df.columns:
            seed_sources.extend(existing_path_from_cell(row.get(c)))

    if "rerun_task_dir" in df.columns and "rerun_run_name" in df.columns:
        task_dir_val = row.get("rerun_task_dir")
        run_name_val = row.get("rerun_run_name")
        if pd.notna(task_dir_val) and pd.notna(run_name_val):
            td = ROOT / str(task_dir_val) if not str(task_dir_val).startswith("/") else Path(str(task_dir_val))
            rn = str(run_name_val).strip()
            if td.exists() and rn:
                p = td / rn
                if p.exists():
                    seed_sources.append(p)

    if "rerun_task_dir" in df.columns:
        task_dir_val = row.get("rerun_task_dir")
        if pd.notna(task_dir_val):
            td = ROOT / str(task_dir_val) if not str(task_dir_val).startswith("/") else Path(str(task_dir_val))
            if td.exists():
                seed_sources.append(td)

    seed_sources = list(dict.fromkeys(seed_sources))

    scanned_files = []
    labels = []
    hits_all = []

    for src in seed_sources:
        for f in candidate_files_from_path(src):
            txt = read_text_safe(f)
            label, hits = classify_text(txt)
            if label != "unknown":
                scanned_files.append(str(f.relative_to(ROOT)) if str(f).startswith(str(ROOT)) else str(f))
                labels.append(label)
                hits_all.extend(hits)

    hits_all = list(dict.fromkeys(hits_all))

    if "benchmark" in labels and "data_new" in labels:
        final_label = "mixed"
    elif "benchmark" in labels:
        final_label = "benchmark"
    elif "data_new" in labels:
        final_label = "data_new"
    else:
        final_label = "unknown"

    rows.append({
        "task": task,
        "provenance_label": final_label,
        "matched_hits": ";".join(hits_all),
        "evidence_files": " | ".join(scanned_files[:20]),
        "rerun_results_file": row.get("rerun_results_file", ""),
        "rerun_task_dir": row.get("rerun_task_dir", ""),
        "rerun_run_name": row.get("rerun_run_name", ""),
    })

out = pd.DataFrame(rows).sort_values(["provenance_label", "task"])
out.to_csv(OUT_CSV, index=False)

print("=== PROVENANCE AUDIT ===")
print(out.to_string(index=False))

print("\n=== COUNTS ===")
print(out["provenance_label"].value_counts(dropna=False).to_string())

print("\nSaved:", OUT_CSV)
