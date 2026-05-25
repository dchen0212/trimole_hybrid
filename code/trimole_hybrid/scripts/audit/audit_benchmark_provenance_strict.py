from pathlib import Path
import pandas as pd
import re

ROOT = Path("/mnt/afs/250010150/zhensheng/trimole")
RESULTS = ROOT / "results/model_log"

FINAL_MASTER = RESULTS / "benchmark_clean_rerun/benchmark_clean_final_master_strict.csv"

OUT_DIR = ROOT / "results/audit_tdc_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "benchmark_provenance_audit_strict.csv"
OUT_TXT = OUT_DIR / "benchmark_provenance_audit_strict.txt"

TASKS = [
    "ames","bbb_martins","bioavailability_ma","caco2_wang","clearance_hepatocyte_az",
    "clearance_microsome_az","cyp2c9_substrate_carbonmangels","cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels","cyp2d6_veith","cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith","dili","half_life_obach","herg","hia_hou","ld50_zhu",
    "lipophilicity_astrazeneca","pgp_broccatelli","ppbr_az","solubility_aqsoldb","vdss_lombardo"
]

# 只看真正可能包含 run 参数 / provenance 的目录
SEARCH_DIRS = [
    RESULTS / "benchmark_clean_rerun/run_20260415_2016",
    RESULTS / "txg_weight_search_benchmark_clean",
    RESULTS / "xg_weight_search_missing_trimole",
    RESULTS / "txg_refine_v2_5tasks",
]

# 只看真正可能包含参数/日志的文件类型
ALLOW_EXTS = {".json", ".yaml", ".yml", ".sh", ".log", ".out", ".txt", ".md"}

# 显式排除会引入误报的汇总文件
EXCLUDE_PATTERNS = [
    "final_master",
    "final_report",
    "report_table",
    "manifest",
    "patch.csv",
    "audit",
    "summary",
    "results_official_metrics.csv",
    "results_all.csv",
]

BENCH_PATTERNS = [
    "data/data_benchmark",
    "./data/data_benchmark",
    "data_benchmark",
]

NEW_PATTERNS = [
    "data/data_new",
    "./data/data_new",
    "data_new",
]

def read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def should_exclude(path: Path):
    s = str(path).lower()
    return any(p in s for p in EXCLUDE_PATTERNS)

def collect_files():
    files = []
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in ALLOW_EXTS:
                continue
            if should_exclude(p):
                continue
            files.append(p)
    return files

def classify_text(text: str):
    t = text.lower()
    has_bench = any(p in t for p in BENCH_PATTERNS)
    has_new = any(p in t for p in NEW_PATTERNS)

    if has_bench and has_new:
        return "mixed"
    if has_bench:
        return "confirmed_benchmark"
    if has_new:
        return "confirmed_nonbenchmark"
    return "unknown"

def merge_status(a, b):
    if a == b:
        return a
    if "mixed" in (a, b):
        return "mixed"
    if a == "unknown":
        return b
    if b == "unknown":
        return a
    return "mixed"

def task_related(task: str, path: Path, text: str):
    tl = task.lower()
    s = str(path).lower()
    t = text.lower()
    return (tl in s) or (tl in t)

def load_final_master():
    return pd.read_csv(FINAL_MASTER)

def main():
    if not FINAL_MASTER.exists():
        raise FileNotFoundError(f"Missing final master: {FINAL_MASTER}")

    files = collect_files()
    final_df = load_final_master()

    rows = []
    for _, r in final_df.iterrows():
        task = str(r["task"]).strip()
        metric_name = r.get("primary_metric_name", "")
        metric_val = r.get("primary_metric", "")
        final_source = r.get("final_source", "")
        final_note = r.get("final_note", "")

        task_status = "unknown"
        evidence = []

        for f in files:
            txt = read_text(f)
            if not task_related(task, f, txt):
                continue
            st = classify_text(txt + "\n" + str(f))
            if st != "unknown":
                evidence.append((str(f), st))
                task_status = merge_status(task_status, st)

        rows.append({
            "task": task,
            "primary_metric_name": metric_name,
            "primary_metric": metric_val,
            "final_source": final_source,
            "benchmark_status": task_status,
            "evidence_count": len(evidence),
            "evidence_paths": " | ".join([f"{p} [{st}]" for p, st in evidence[:20]]),
            "final_note": final_note,
        })

    out = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("=== strict benchmark provenance audit ===\n")
        f.write(out.to_string(index=False))
        f.write("\n\n=== status counts ===\n")
        f.write(out["benchmark_status"].value_counts(dropna=False).to_string())
        f.write("\n")

    print("Saved:", OUT_CSV)
    print("Saved:", OUT_TXT)
    print()
    print(out.to_string(index=False))
    print()
    print("=== status counts ===")
    print(out["benchmark_status"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
