from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path("<PROJECT_ROOT>/trimole")
FINAL_CSV = ROOT / "results/model_log/final_validation_selected_submission/final_validation_selected_submission.csv"
OUT_CSV = ROOT / "results/model_log/final_validation_selected_submission/final_submission_provenance_audit_v2.csv"

TARGET_META_NAMES = [
    "00_run_meta.txt",
    "run_args.json",
    "run_args_fixed.json",
    "task_summary.json",
]

TARGET_GLOBS = [
    "*.log",
    "*.jsonl",
    "*.txt",
    "*.json",
]

POS_BENCH = re.compile(r"data/data_benchmark|data_benchmark")
POS_NEW   = re.compile(r"data/data_new|data_new")

def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def classify_text(txt: str) -> tuple[str, list[str]]:
    has_bench = bool(POS_BENCH.search(txt))
    has_new   = bool(POS_NEW.search(txt))
    hits = []
    if has_bench:
        hits.append("data_benchmark")
    if has_new:
        hits.append("data_new")
    if has_bench and has_new:
        return "mixed", hits
    if has_bench:
        return "confirmed_benchmark", hits
    if has_new:
        return "confirmed_nonbenchmark", hits
    return "unknown", hits

def normalize_cell_to_paths(cell) -> list[Path]:
    out = []
    if pd.isna(cell):
        return out
    s = str(cell).strip()
    if not s:
        return out

    parts = [x.strip() for x in s.split(" + ")]
    for part in parts:
        if "::" in part:
            continue
        p = ROOT / part if not part.startswith("/") else Path(part)
        if p.exists():
            out.append(p)
    return out

def gather_search_roots(row) -> list[Path]:
    roots = []

    for c in ["rerun_results_file", "rerun_task_dir"]:
        if c in row.index:
            roots.extend(normalize_cell_to_paths(row[c]))

    if "rerun_task_dir" in row.index and "rerun_run_name" in row.index:
        td = row.get("rerun_task_dir")
        rn = row.get("rerun_run_name")
        if pd.notna(td) and pd.notna(rn):
            task_dir = ROOT / str(td) if not str(td).startswith("/") else Path(str(td))
            run_name = str(rn).strip()
            if task_dir.exists() and run_name:
                run_dir = task_dir / run_name
                if run_dir.exists():
                    roots.append(run_dir)

    # rerun_results_file parent dirs
    if "rerun_results_file" in row.index:
        for p in normalize_cell_to_paths(row["rerun_results_file"]):
            roots.append(p.parent if p.is_file() else p)

    # fallback: final_best_v4_runs/<task>
    task = str(row["task"])
    fb = ROOT / "results/model_log/final_best_v4_runs" / task
    if fb.exists():
        roots.append(fb)

    # dedup
    uniq = []
    seen = set()
    for p in roots:
        s = str(p.resolve()) if p.exists() else str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq

def scan_root_deep(root: Path, max_files=300):
    found = []
    if not root.exists():
        return found

    candidates = []

    if root.is_file():
        candidates.append(root)
        root = root.parent

    # exact meta files within depth 4
    for name in TARGET_META_NAMES:
        candidates.extend(list(root.rglob(name)))

    # logs/json/txt within depth-ish by limiting total count later
    for pat in TARGET_GLOBS:
        candidates.extend(list(root.rglob(pat)))

    # dedup + sort by shallower path first
    uniq = []
    seen = set()
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen and p.is_file():
            seen.add(key)
            uniq.append(p)

    uniq = sorted(uniq, key=lambda p: (len(p.parts), str(p)))[:max_files]

    for f in uniq:
        txt = read_text_safe(f)
        label, hits = classify_text(txt)
        if label != "unknown":
            found.append((f, label, hits))
    return found

df = pd.read_csv(FINAL_CSV)

rows = []
for _, row in df.iterrows():
    task = str(row["task"])
    roots = gather_search_roots(row)

    evidence = []
    labels = []
    hits_all = []

    for r in roots:
        for f, label, hits in scan_root_deep(r):
            evidence.append(f)
            labels.append(label)
            hits_all.extend(hits)

    hits_all = list(dict.fromkeys(hits_all))

    if "mixed" in labels or ("confirmed_benchmark" in labels and "confirmed_nonbenchmark" in labels):
        final_label = "mixed"
    elif "confirmed_benchmark" in labels:
        final_label = "confirmed_benchmark"
    elif "confirmed_nonbenchmark" in labels:
        final_label = "confirmed_nonbenchmark"
    else:
        final_label = "unknown"

    evidence_str = []
    for f in evidence[:20]:
        try:
            evidence_str.append(str(f.relative_to(ROOT)))
        except Exception:
            evidence_str.append(str(f))

    rows.append({
        "task": task,
        "provenance_label": final_label,
        "matched_hits": ";".join(hits_all),
        "evidence_files": " | ".join(evidence_str),
        "rerun_results_file": row.get("rerun_results_file", ""),
        "rerun_task_dir": row.get("rerun_task_dir", ""),
        "rerun_run_name": row.get("rerun_run_name", ""),
    })

out = pd.DataFrame(rows).sort_values(["provenance_label", "task"])
out.to_csv(OUT_CSV, index=False)

print("=== PROVENANCE AUDIT V2 ===")
print(out.to_string(index=False))

print("\n=== COUNTS ===")
print(out["provenance_label"].value_counts(dropna=False).to_string())

print("\nSaved:", OUT_CSV)
