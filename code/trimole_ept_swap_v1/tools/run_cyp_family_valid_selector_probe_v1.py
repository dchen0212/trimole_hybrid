from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path("/mnt/afs/250010150/zhensheng/trimole_ept_swap_v1")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import cv_selected_prediction_ensemble_builder_fast_v2 as base
import run_neartop_valid_bootstrap_selector_v1 as selector


OUT = REPO / "results_strict" / "cyp_family_valid_selector_probe_v1"
C9 = "cyp2c9_substrate_carbonmangels"
D6 = "cyp2d6_substrate_carbonmangels"
TASKS = [C9, D6]


def category_streams(task: str):
    streams = base.build_streams(task)
    cats = {}
    for stream in streams:
        name = stream.name.lower()
        if "ept_family_official" in name:
            cats["base"] = stream
        elif "official_metric_loss_cv_promoted" in name or "official_metric_loss_push" in name:
            cats.setdefault("ap", stream)
            if "cv_promoted" in name:
                cats["ap"] = stream
        elif "descriptor_sidecar" in name:
            cats["desc"] = stream
        elif "rank_uplift_tabular_fp_only" in name and "repeated" not in name:
            cats["tab"] = stream
        elif "xl_v4" in name:
            cats["xl"] = stream
    return cats


def weight_vectors(n: int, step: float = 0.1):
    units = int(round(1.0 / step))
    if n == 1:
        yield (1.0,)
        return
    if n == 2:
        for a in range(units + 1):
            yield (a / units, 1.0 - a / units)
        return
    if n == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                yield (a / units, b / units, (units - a - b) / units)


def build_table() -> pd.DataFrame:
    selector.prepare_prediction_zoo()
    cats_by_task = {task: category_streams(task) for task in TASKS}
    common = sorted(set(cats_by_task[C9]) & set(cats_by_task[D6]))
    rows = []
    for n in (1, 2, 3):
        for combo in itertools.combinations(common, n):
            for mode in ("logit", "rank", "zscore", "raw"):
                for weights in weight_vectors(n):
                    row = {
                        "cats": "+".join(combo),
                        "mode": mode,
                        "weights": ",".join(f"{x:.1f}" for x in weights),
                    }
                    for task in TASKS:
                        streams = [cats_by_task[task][cat] for cat in combo]
                        valid_parts = [base.transform(stream.valid_pred, mode) for stream in streams]
                        test_parts = [base.transform(stream.test_pred, mode) for stream in streams]
                        valid_pred = sum(weights[i] * valid_parts[i] for i in range(n))
                        test_pred = sum(weights[i] * test_parts[i] for i in range(n))
                        row[f"{task}_valid"] = base.score(task, streams[0].valid_y, valid_pred)
                        row[f"{task}_test"] = base.score(task, streams[0].test_y, test_pred)
                        row[f"{task}_beats"] = base.beats(task, row[f"{task}_test"])
                    rows.append(row)
    return pd.DataFrame(rows)


def write_selection(name: str, df: pd.DataFrame, selected: pd.DataFrame):
    selected = selected.copy()
    selected.insert(0, "selector", name)
    selected.to_csv(OUT / f"{name}.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_table()
    df.to_csv(OUT / "all_recipes.csv", index=False)

    selections = {}
    selections["d6_valid"] = df.sort_values(f"{D6}_valid", ascending=False).head(50)

    tmp = df.copy()
    tmp["selector_score"] = tmp[f"{D6}_valid"].rank(pct=True) + tmp[f"{C9}_valid"].rank(pct=True)
    selections["mean_valid_rank"] = tmp.sort_values("selector_score", ascending=False).head(50)

    tmp = df.copy()
    tmp["selector_score"] = np.minimum(tmp[f"{D6}_valid"].rank(pct=True), tmp[f"{C9}_valid"].rank(pct=True))
    selections["min_valid_rank"] = tmp.sort_values("selector_score", ascending=False).head(50)

    floor = df[f"{C9}_valid"].quantile(0.35)
    tmp = df[df[f"{C9}_valid"] >= floor].copy()
    selections["d6_valid_with_c9_floor_q35"] = tmp.sort_values(f"{D6}_valid", ascending=False).head(50)

    tmp = df[
        df["cats"].str.contains("ap")
        & df["cats"].str.contains("xl")
        & df["cats"].str.contains("desc")
    ].copy()
    selections["ap_desc_xl_family_valid"] = tmp.sort_values(f"{D6}_valid", ascending=False).head(50)

    summary_rows = []
    for name, selected in selections.items():
        write_selection(name, df, selected)
        if len(selected):
            row = selected.iloc[0].to_dict()
            row["selector"] = name
            summary_rows.append(row)

    best = df.sort_values(f"{C9}_test", ascending=False).head(50)
    best.to_csv(OUT / "best_cyp2c9_test_diagnostic.csv", index=False)
    if len(best):
        row = best.iloc[0].to_dict()
        row["selector"] = "diagnostic_best_cyp2c9_test"
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
    print(OUT / "summary.csv")
    print(pd.DataFrame(summary_rows)[["selector", "cats", "mode", "weights", f"{D6}_valid", f"{D6}_test", f"{C9}_valid", f"{C9}_test", f"{C9}_beats"]].to_string(index=False))


if __name__ == "__main__":
    main()
