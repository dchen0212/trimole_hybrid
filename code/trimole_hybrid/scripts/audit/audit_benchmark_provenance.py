from pathlib import Path
import pandas as pd
import json
import re

ROOT = Path("<PROJECT_ROOT>/trimole")
RESULTS = ROOT / "results/model_log"
BENCH_DIR = str((ROOT / "data/data_benchmark").resolve())
NEW_DIR = str((ROOT / "data/data_new").resolve())

FINAL_MASTER = RESULTS / "benchmark_clean_rerun/benchmark_clean_final_master_strict.csv"
BASE_OFFICIAL = RESULTS / "benchmark_clean_rerun/run_20260415_2016/results_official_metrics.csv"
V2_PATCH = RESULTS / "benchmark_clean_rerun/benchmark_clean_v2_patch.csv"
SELECTIVE_PATCH = RESULTS / "benchmark_clean_rerun/benchmark_clean_txg_selective_patch.csv"

OUT_DIR = ROOT / "results/audit_tdc_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "benchmark_provenance_audit.csv"
OUT_TXT = OUT_DIR / "benchmark_provenance_audit.txt"

TASKS = [
    "ames","bbb_martins","bioavailability_ma","caco2_wang","clearance_hepatocyte_az",
    "clearance_microsome_az","cyp2c9_substrate_carbonmangels","cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels","cyp2d6_veith","cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith","dili","half_life_obach","herg","hia_hou","ld50_zhu",
    "lipophilicity_astrazeneca","pgp_broccatelli","ppbr_az","solubility_aqsoldb","vdss_lombardo"
]

SEARCH_DIRS = [
    RESULTS / "benchmark_clean_rerun",
    RESULTS / "txg_weight_search_benchmark_clean",
    RESULTS / "xg_weight_search_missing_trimole",
    RESULTS / "txg_refine_v2_5tasks",
    RESULTS,
]

TEXT_EXTS = {".txt", ".log", ".json", ".yaml", ".yml", ".csv", ".md", ".sh", ".out"}

def read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def collect_candidate_files():
    files = []
    seen = set()
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in TEXT_EXTS:
                continue
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                files.append(p)
    return files

def classify_text(text: str):
    t = text.lower()
    has_bench = ("data/data_benchmark" in t) or ("data_benchmark" in t and "data_new" not in t)
    has_new = ("data/data_new" in t) or ("data_new" in t)
    if has_bench and has_new:
        return "mixed"
    if has_bench:
        return "confirmed_benchmark"
    if has_new:
        return "confirmed_nonbenchmark"
    return "unknown"

def merge_status(a, b):
    order = ["unknown", "confirmed_benchmark", "confirmed_nonbenchmark", "mixed"]
    if a == b:
        return a
    if "mixed" in (a, b):
        return "mixed"
    if a == "unknown":
        return b
    if b == "unknown":
        return a
    if a != b:
        return "mixed"
    return a

def build_task_index(files):
    task_hits = {t: [] for t in TASKS}
    for f in files:
        txt = read_text(f)
        low = txt.lower()
        fname = f.name.lower()
        pathlow = str(f).lower()
        for task in TASKS:
            if task.lower() in low or task.lower() in fname or task.lower() in pathlow:
                status = classify_text(txt + "\n" + pathlow)
                task_hits[task].append({
                    "file": str(f),
                    "status": status
                })
    return task_hits

def load_final_master():
    return pd.read_csv(FINAL_MASTER)

def determine_expected_source(df):
    out = {}
    for _, r in df.iterrows():
        out[str(r["task"]).strip()] = {
            "final_source": r.get("final_source", ""),
            "primary_metric_name": r.get("primary_metric_name", ""),
            "primary_metric": r.get("primary_metric", ""),
            "final_note": r.get("final_note", ""),
        }
    return out

def derive_final_status(task, final_source, hits):
    status = "unknown"
    evidence_files = []

    # 先汇总 task 命中的证据
    for h in hits:
        status = merge_status(status, h["status"])
        evidence_files.append(h["file"])

    # 对 base / v2 / manual 做规则加权，但不替代文本证据
    if final_source == "base_official_metrics":
        # 基础表来自 benchmark_clean_rerun 正式线，但如果没有直接文字证据，仍保持 unknown/benchmark 倾向
        if status == "unknown":
            status = "unknown"
    elif final_source == "v2_patch":
        if status == "unknown":
            status = "unknown"
    elif final_source == "manual_keep_old_best":
        if status == "unknown":
            status = "unknown"

    return status, evidence_files[:10]

def main():
    if not FINAL_MASTER.exists():
        raise FileNotFoundError(f"Missing final master: {FINAL_MASTER}")

    files = collect_candidate_files()
    task_hits = build_task_index(files)
    final_df = load_final_master()
    source_map = determine_expected_source(final_df)

    rows = []
    for task in TASKS:
        meta = source_map.get(task, {})
        final_source = meta.get("final_source", "")
        metric_name = meta.get("primary_metric_name", "")
        metric_val = meta.get("primary_metric", "")
        final_note = meta.get("final_note", "")
        hits = task_hits.get(task, [])

        final_status, evidence_files = derive_final_status(task, final_source, hits)

        rows.append({
            "task": task,
            "primary_metric_name": metric_name,
            "primary_metric": metric_val,
            "final_source": final_source,
            "benchmark_status": final_status,
            "evidence_count": len(hits),
            "evidence_paths": " | ".join(evidence_files),
            "final_note": final_note,
        })

    out = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("=== benchmark provenance audit ===\n")
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
