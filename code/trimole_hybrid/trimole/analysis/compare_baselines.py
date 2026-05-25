from __future__ import annotations

import argparse
import json
from datetime import datetime
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PRIMARY_METRIC_CANDIDATES = [
    "AUROC",
    "AUC",
    "AUPRC",
    "AUCPR",
    "MAE",
    "RMSE",
    "MSE",
    "ACC",
    "Accuracy",
    "Spearman",
    "SPEARMAN",
]


LOWER_IS_BETTER_METRICS = {"MAE", "RMSE", "MSE"}


def _infer_dataset_id_from_filename(stem: str) -> str:
    s = stem.strip()
    s = re.sub(r"^tdcommons_", "", s, flags=re.IGNORECASE)
    # Common patterns:
    # - tdcommons_amesleaderboard_leaderboard
    # - tdcommons_bioavailability_maleaderboard_leaderboard
    s = re.sub(r"_?leaderboard(_leaderboard)?$", "", s, flags=re.IGNORECASE)
    # Remove a remaining trailing 'leaderboard' glued to dataset name
    s = re.sub(r"leaderboard$", "", s, flags=re.IGNORECASE)
    return s


def _normalize_task_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"leaderboard", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _parse_score_cell(x: object) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    # common format: 0.456±0.008
    s = s.replace("±", "+-")
    s = s.split("+-", 1)[0]
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return float("nan")


@dataclass
class BaselineEntry:
    task_norm: str
    dataset_id: str
    metric_name: str
    baseline_best: float
    baseline_model: str


def load_baselines(baselines_dir: Path) -> Dict[str, BaselineEntry]:
    baselines: Dict[str, BaselineEntry] = {}

    for csv_path in sorted(baselines_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Identify metric column.
        metric_col = None
        for c in df.columns:
            if str(c).strip() in PRIMARY_METRIC_CANDIDATES:
                metric_col = str(c).strip()
                break
        if metric_col is None:
            # skip unrecognized schema
            continue

        dataset_id = ""
        if "dataset_id" in df.columns:
            dataset_id = str(df.iloc[0]["dataset_id"])
        else:
            dataset_id = _infer_dataset_id_from_filename(csv_path.stem)

        # Best baseline is Rank==1 if present, else first row.
        row = df.iloc[0]
        if "Rank" in df.columns:
            try:
                df_rank = df.copy()
                df_rank["Rank"] = pd.to_numeric(df_rank["Rank"], errors="coerce")
                df_rank = df_rank.sort_values("Rank", ascending=True)
                row = df_rank.iloc[0]
            except Exception:
                row = df.iloc[0]

        baseline_best = _parse_score_cell(row.get(metric_col))
        baseline_model = str(row.get("Model", ""))

        task_norm = _normalize_task_name(dataset_id)
        baselines[task_norm] = BaselineEntry(
            task_norm=task_norm,
            dataset_id=dataset_id,
            metric_name=metric_col,
            baseline_best=baseline_best,
            baseline_model=baseline_model,
        )

    return baselines


@dataclass
class Leaderboard:
    task_norm: str
    dataset_id: str
    metric_name: str
    higher_is_better: bool
    topk: pd.DataFrame  # columns: Rank(int), Model(str), Score(float)


def _pick_metric_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip() in PRIMARY_METRIC_CANDIDATES:
            return str(c).strip()
    return None


def _load_leaderboard_csv(csv_path: Path, topk: int) -> Optional[Leaderboard]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    metric_col = _pick_metric_column(df)
    if metric_col is None:
        return None

    if "dataset_id" in df.columns:
        dataset_id = str(df.iloc[0]["dataset_id"])
    else:
        dataset_id = _infer_dataset_id_from_filename(csv_path.stem)

    d = df.copy()
    if "Rank" in d.columns:
        d["Rank"] = pd.to_numeric(d["Rank"], errors="coerce")
    else:
        d["Rank"] = np.arange(1, len(d) + 1, dtype=int)

    if "Model" not in d.columns:
        # schema mismatch
        return None

    d["Score"] = d[metric_col].map(_parse_score_cell)
    d = d.sort_values("Rank", ascending=True, na_position="last")
    d = d[["Rank", "Model", "Score"]].copy()

    # Keep top-k rows with numeric scores if possible, but don't drop everything.
    d_nonan = d.dropna(subset=["Score"])
    if not d_nonan.empty:
        d = d_nonan
    if topk > 0:
        d = d.head(topk)

    task_norm = _normalize_task_name(dataset_id)
    higher_is_better = metric_col.upper() not in LOWER_IS_BETTER_METRICS

    return Leaderboard(
        task_norm=task_norm,
        dataset_id=dataset_id,
        metric_name=metric_col,
        higher_is_better=higher_is_better,
        topk=d,
    )


def load_leaderboards(baselines_dir: Path, topk: int = 5) -> Dict[str, Leaderboard]:
    lbs: Dict[str, Leaderboard] = {}
    for csv_path in sorted(baselines_dir.glob("*.csv")):
        lb = _load_leaderboard_csv(csv_path, topk=topk)
        if lb is None:
            continue
        lbs[lb.task_norm] = lb
    return lbs


def _find_latest_run_dir(path: Path) -> Path:
    """Accept either:
    - results/model_log/<outer_run>/run_YYYY...
    - results/model_log/<outer_run>
    - results/model_log/run_YYYY...

    Returns the run_YYYY... directory.
    """
    p = path.resolve()
    if p.name.startswith("run_") and p.is_dir():
        return p

    candidates = [d for d in p.glob("run_*") if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run_* directory under: {p}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def load_ours(run_root: Path) -> pd.DataFrame:
    run_dir = _find_latest_run_dir(run_root)
    results_csv = run_dir / "results_all.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)

        # Some older/partial summaries may miss newer metric columns (e.g. test_spearman).
        # In that case, enrich from each task's meta.json so leaderboard comparisons work.
        fill_keys = [
            "task_type",
            "primary_metric_name",
            "primary_metric",
            "best_valid_primary",
            "best_epoch",
            "device",
            "seed",
            "test_acc",
            "test_auc",
            "test_auprc",
            "test_mae",
            "test_rmse",
            "test_spearman",
        ]

        meta_by_task_norm: dict[str, dict] = {}
        for task_dir in sorted(run_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            meta_path = task_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            task_name = str(meta.get("task") or task_dir.name)
            meta_by_task_norm[_normalize_task_name(task_name)] = meta

        # Ensure columns exist.
        for k in fill_keys:
            if k not in df.columns:
                df[k] = np.nan

        # Fill NaNs from meta.json values.
        for idx, row in df.iterrows():
            task_name = str(row.get("task", ""))
            meta = meta_by_task_norm.get(_normalize_task_name(task_name))
            if not meta:
                continue
            for k in fill_keys:
                v = meta.get(k)
                if v is None:
                    continue
                if pd.isna(row.get(k)):
                    df.at[idx, k] = v
    else:
        # fallback: collect meta.json
        rows = []
        for task_dir in sorted(run_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            meta_path = task_dir / "meta.json"
            if meta_path.exists():
                rows.append(pd.read_json(meta_path.open("r"), typ="series"))
        df = pd.DataFrame(rows)

    if "task" not in df.columns:
        raise ValueError(f"Missing column 'task' in {results_csv}")

    df["task_norm"] = df["task"].astype(str).map(_normalize_task_name)
    return df


def compute_comparison(ours: pd.DataFrame, baselines: Dict[str, BaselineEntry]) -> pd.DataFrame:
    rows = []
    for _, r in ours.iterrows():
        task = str(r.get("task"))
        task_norm = str(r.get("task_norm"))

        b = baselines.get(task_norm)
        metric_name = str(r.get("primary_metric_name", ""))
        our_val = float(pd.to_numeric(r.get("primary_metric"), errors="coerce")) if r.get("primary_metric") is not None else float("nan")

        baseline_metric = None
        baseline_best = float("nan")
        baseline_model = ""
        dataset_id = ""
        if b is not None:
            baseline_metric = b.metric_name
            baseline_best = float(b.baseline_best)
            baseline_model = b.baseline_model
            dataset_id = b.dataset_id

        # If our primary metric differs from baseline metric, try to pick matching one.
        # (e.g. legacy runs before regression support)
        metric_for_compare = baseline_metric or metric_name
        our_for_compare = our_val
        if metric_for_compare:
            m = metric_for_compare.upper()
            if m in {"AUROC", "AUC"}:
                our_for_compare = float(pd.to_numeric(r.get("test_auc"), errors="coerce"))
            elif m in {"AUPRC", "AUCPR"}:
                our_for_compare = float(pd.to_numeric(r.get("test_auprc"), errors="coerce"))
            elif m == "MAE":
                our_for_compare = float(pd.to_numeric(r.get("test_mae"), errors="coerce"))
            elif m == "RMSE":
                our_for_compare = float(pd.to_numeric(r.get("test_rmse"), errors="coerce"))
            elif m in {"ACC", "ACCURACY"}:
                our_for_compare = float(pd.to_numeric(r.get("test_acc"), errors="coerce"))
            elif m == "SPEARMAN":
                our_for_compare = float(pd.to_numeric(r.get("test_spearman"), errors="coerce"))

        higher_is_better = True
        if metric_for_compare and metric_for_compare.upper() in LOWER_IS_BETTER_METRICS:
            higher_is_better = False

        if np.isnan(our_for_compare) or np.isnan(baseline_best):
            delta = float("nan")
        else:
            delta = (our_for_compare - baseline_best) if higher_is_better else (baseline_best - our_for_compare)

        rows.append(
            {
                "task": task,
                "dataset_id": dataset_id,
                "metric": metric_for_compare,
                "baseline_best": baseline_best,
                "baseline_model": baseline_model,
                "our": our_for_compare,
                "improvement": delta,
                "direction": "higher_better" if higher_is_better else "lower_better",
                "task_type": r.get("task_type"),
            }
        )

    df = pd.DataFrame(rows)
    # Sort: biggest improvement first
    df = df.sort_values("improvement", ascending=False, na_position="last")
    return df


def _our_value_for_metric(row: pd.Series, metric_name: str) -> float:
    if not metric_name:
        return float("nan")
    m = metric_name.upper()
    if m in {"AUROC", "AUC"}:
        return float(pd.to_numeric(row.get("test_auc"), errors="coerce"))
    if m in {"AUPRC", "AUCPR"}:
        return float(pd.to_numeric(row.get("test_auprc"), errors="coerce"))
    if m == "MAE":
        return float(pd.to_numeric(row.get("test_mae"), errors="coerce"))
    if m == "RMSE":
        return float(pd.to_numeric(row.get("test_rmse"), errors="coerce"))
    if m in {"ACC", "ACCURACY"}:
        return float(pd.to_numeric(row.get("test_acc"), errors="coerce"))
    if m == "SPEARMAN":
        return float(pd.to_numeric(row.get("test_spearman"), errors="coerce"))
    # fallback to primary_metric
    return float(pd.to_numeric(row.get("primary_metric"), errors="coerce"))


def _safe_filename(s: str) -> str:
    s2 = _normalize_task_name(s)
    return s2 or "unknown"


def _render_task_html(task: str, metric: str, rows: List[dict], out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    scores = [r.get("score") for r in rows]
    scores_num = [float(x) for x in scores if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if scores_num:
        vmin = float(min(scores_num))
        vmax = float(max(scores_num))
        span = max(vmax - vmin, 1e-12)
    else:
        vmin, vmax, span = 0.0, 1.0, 1.0

    def bar_cell(v: object, is_ours: bool) -> str:
        try:
            x = float(v)
        except Exception:
            return ""
        if np.isnan(x):
            return ""
        width = int(min(100, max(0, (x - vmin) / span * 100)))
        color = "#1f77b4" if is_ours else "#888"
        return (
            "<div style='display:flex;align-items:center;gap:10px'>"
            "<div style='width:220px;background:#f2f2f2;height:10px;border-radius:6px;overflow:hidden'>"
            f"<div style='width:{width}%;background:{color};height:10px'></div>"
            "</div>"
            f"<span style='font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace'>{x:.6g}</span>"
            "</div>"
        )

    trs = []
    for r in rows:
        kind = str(r.get("kind"))
        is_ours = kind == "ours"
        rank = r.get("rank")
        rank_txt = "" if rank is None or (isinstance(rank, float) and np.isnan(rank)) else str(int(rank))
        model = str(r.get("model", ""))
        score = r.get("score")
        trs.append(
            "<tr>"
            f"<td>{rank_txt}</td>"
            f"<td>{'Ours' if is_ours else 'Leaderboard'}</td>"
            f"<td style='font-weight:{'700' if is_ours else '400'}'>{model}</td>"
            f"<td>{bar_cell(score, is_ours=is_ours)}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>{task} — Top leaderboard vs Ours</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 6px 0; }}
    .sub {{ color:#555; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 10px 8px; vertical-align: middle; }}
    th {{ text-align: left; position: sticky; top: 0; background: #fff; z-index: 1; }}
  </style>
</head>
<body>
  <h1>{task}</h1>
  <div class='sub'>metric: <b>{metric}</b>（条形长度按该 task 内 min/max 线性缩放）</div>
  <table>
    <thead>
      <tr><th>Rank</th><th>Source</th><th>Model</th><th>Score</th></tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</body>
</html>"""

    out_html.write_text(html, encoding="utf-8")


def _try_plot_task(task: str, metric: str, rows: List[dict], higher_is_better: bool, out_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is None:
        # Fallback to Pillow (no extra deps; already in env.yml).
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return False

        ordered = [rr for rr in rows if rr.get("kind") == "leaderboard"]
        ordered.sort(key=lambda rr: (rr.get("rank") is None, rr.get("rank") or 1e9))
        ordered += [rr for rr in rows if rr.get("kind") == "ours"]

        vals = []
        for rr in ordered:
            try:
                v = float(rr.get("score"))
            except Exception:
                v = float("nan")
            if not np.isnan(v):
                vals.append(v)
        if not vals:
            return False

        vmin = float(min(vals))
        vmax = float(max(vals))
        span = max(vmax - vmin, 1e-12)

        # Layout (in pixels)
        width = 1400
        pad = 20
        title_h = 54
        row_h = 34
        bar_h = 12
        label_w = 520
        bar_w = 620
        val_w = width - pad * 2 - label_w - bar_w
        height = pad * 2 + title_h + row_h * len(ordered)

        def _load_font(size: int) -> ImageFont.FreeTypeFont:
            try:
                return ImageFont.truetype("DejaVuSans.ttf", size=size)
            except Exception:
                return ImageFont.load_default()

        font_title = _load_font(18)
        font_body = _load_font(14)

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        draw.text((pad, pad), f"{task} — Top leaderboard vs Ours", fill="#111111", font=font_title)
        draw.text((pad, pad + 26), f"metric: {metric}", fill="#555555", font=font_body)

        y0 = pad + title_h
        for i, rr in enumerate(ordered):
            kind = str(rr.get("kind"))
            model = str(rr.get("model", ""))
            rank = rr.get("rank")
            if kind == "leaderboard":
                prefix = "" if rank is None else f"#{int(rank)} "
                label = prefix + model
            else:
                label = "Ours (Trimole)"

            # truncate long labels
            if len(label) > 60:
                label = label[:57] + "..."

            try:
                v = float(rr.get("score"))
            except Exception:
                v = float("nan")

            y = y0 + i * row_h
            draw.text((pad, y + 6), label, fill="#111111", font=font_body)

            bar_x = pad + label_w
            bar_y = y + (row_h - bar_h) // 2
            draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=6, fill="#f2f2f2", outline=None)

            if not np.isnan(v):
                pct = float((v - vmin) / span)
                pct = max(0.0, min(1.0, pct))
                w = int(bar_w * pct)
                color = "#1f77b4" if kind == "ours" else "#9e9e9e"
                if w > 0:
                    draw.rounded_rectangle((bar_x, bar_y, bar_x + w, bar_y + bar_h), radius=6, fill=color, outline=None)

            val_x = pad + label_w + bar_w + 14
            val_txt = "" if np.isnan(v) else f"{v:.6g}"
            draw.text((val_x, y + 6), val_txt, fill="#111111", font=font_body)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_png)
        return True

    labels = []
    values = []
    colors = []
    for r in rows:
        labels.append(str(r.get("model", "")))
        v = r.get("score")
        try:
            values.append(float(v))
        except Exception:
            values.append(float("nan"))
        colors.append("#1f77b4" if r.get("kind") == "ours" else "#9e9e9e")

    # Keep numeric-only for plotting order, but preserve labels.
    idx = [i for i, v in enumerate(values) if not np.isnan(v)]
    if not idx:
        return False

    # Sort bars so leaderboard ranks are visible (best on top) + ours at bottom.
    # We keep our bar last for readability.
    ours_idx = [i for i, r in enumerate(rows) if r.get("kind") == "ours"]
    ours_idx = ours_idx[0] if ours_idx else None

    base_idx = [i for i in idx if i != ours_idx]
    base_idx.sort(key=lambda i: values[i], reverse=higher_is_better)
    plot_idx = base_idx + ([ours_idx] if ours_idx is not None and ours_idx in idx else [])

    plot_labels = [labels[i] for i in plot_idx]
    plot_values = [values[i] for i in plot_idx]
    plot_colors = [colors[i] for i in plot_idx]

    plt.figure(figsize=(10, max(3.5, 0.5 * len(plot_idx) + 1)))
    plt.barh(plot_labels, plot_values, color=plot_colors)
    plt.gca().invert_yaxis()
    plt.title(f"{task}: Top leaderboard vs Ours")
    plt.xlabel(metric)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    return True


def _try_plot_all_tasks_png(
    summary: pd.DataFrame,
    rows_by_anchor: Dict[str, List[dict]],
    out_png: Path,
    topk: int,
) -> bool:
    """Render a single large PNG that contains all tasks (each task is one subplot).

    This is the non-interactive alternative to all_tasks.html.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if summary.empty:
        return False

    # Stable order: follow summary ordering.
    items: List[Tuple[str, pd.Series]] = []
    for _, sr in summary.iterrows():
        anchor = str(sr.get("anchor") or "")
        if not anchor:
            continue
        rows = rows_by_anchor.get(anchor)
        if not rows:
            continue
        items.append((anchor, sr))

    if not items:
        return False

    if plt is None:
        # Fallback: render a big stacked PNG using Pillow.
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return False

        def _load_font(size: int) -> ImageFont.FreeTypeFont:
            try:
                return ImageFont.truetype("DejaVuSans.ttf", size=size)
            except Exception:
                return ImageFont.load_default()

        font_title = _load_font(20)
        font_task = _load_font(16)
        font_body = _load_font(14)

        width = 2200
        pad = 24
        header_h = 62
        section_pad_y = 14
        row_h = 34
        bar_h = 12
        label_w = 860
        bar_w = 800

        # Compute height
        total_h = pad * 2 + header_h
        section_heights: List[int] = []
        ordered_rows_by_anchor: Dict[str, List[dict]] = {}
        for anchor, sr in items:
            task_rows = rows_by_anchor.get(anchor, [])
            lb_rows = [rr for rr in task_rows if rr.get("kind") == "leaderboard"]
            lb_rows.sort(key=lambda rr: (rr.get("rank") is None, rr.get("rank") or 1e9))
            ours_rows = [rr for rr in task_rows if rr.get("kind") == "ours"]
            ordered = lb_rows + ours_rows
            ordered_rows_by_anchor[anchor] = ordered
            h = section_pad_y * 2 + 28 + row_h * len(ordered)
            section_heights.append(h)
            total_h += h

        img = Image.new("RGB", (width, total_h), "white")
        draw = ImageDraw.Draw(img)

        draw.text((pad, pad), f"Trimole vs TDCommons Leaderboard Top{topk}", fill="#111111", font=font_title)
        draw.text((pad, pad + 30), "One-shot PNG (no HTML). Each block is one dataset.", fill="#555555", font=font_body)

        y = pad + header_h
        for (anchor, sr), sec_h in zip(items, section_heights):
            task = str(sr.get("task", anchor))
            metric = str(sr.get("metric", ""))
            beat = str(sr.get("beat", ""))
            vs = str(sr.get("vs", ""))

            # Title line
            title = f"{task}  ·  {metric}"
            meta = []
            if beat:
                meta.append(f"beat={beat}")
            if vs:
                meta.append(f"vs={vs}")
            if meta:
                title += "  ·  " + "  ".join(meta)

            draw.text((pad, y + section_pad_y), title, fill="#111111", font=font_task)

            rows = ordered_rows_by_anchor.get(anchor, [])
            vals = []
            for rr in rows:
                try:
                    v = float(rr.get("score"))
                except Exception:
                    v = float("nan")
                if not np.isnan(v):
                    vals.append(v)
            vmin = float(min(vals)) if vals else 0.0
            vmax = float(max(vals)) if vals else 1.0
            span = max(vmax - vmin, 1e-12)

            y_rows = y + section_pad_y + 28
            for i, rr in enumerate(rows):
                kind = str(rr.get("kind"))
                model = str(rr.get("model", ""))
                rank = rr.get("rank")
                if kind == "leaderboard":
                    prefix = "" if rank is None else f"#{int(rank)} "
                    label = prefix + model
                else:
                    label = "Ours (Trimole)"

                if len(label) > 95:
                    label = label[:92] + "..."

                try:
                    v = float(rr.get("score"))
                except Exception:
                    v = float("nan")

                yy = y_rows + i * row_h
                draw.text((pad, yy + 6), label, fill="#111111", font=font_body)

                bar_x = pad + label_w
                bar_y = yy + (row_h - bar_h) // 2
                draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=6, fill="#f2f2f2", outline=None)

                if not np.isnan(v):
                    pct = float((v - vmin) / span)
                    pct = max(0.0, min(1.0, pct))
                    w = int(bar_w * pct)
                    color = "#1f77b4" if kind == "ours" else "#9e9e9e"
                    if w > 0:
                        draw.rounded_rectangle((bar_x, bar_y, bar_x + w, bar_y + bar_h), radius=6, fill=color, outline=None)

                val_x = bar_x + bar_w + 16
                val_txt = "" if np.isnan(v) else f"{v:.6g}"
                draw.text((val_x, yy + 6), val_txt, fill="#111111", font=font_body)

            # Divider
            y += sec_h
            draw.line((pad, y - 2, width - pad, y - 2), fill="#eeeeee", width=2)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_png)
        return True

    # Visual tuning for a tall figure.
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    n = len(items)
    # Each task roughly needs ~2.2 inches for topk=5 (6 bars) to keep labels readable.
    fig_h = max(6.0, 2.2 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14.5, fig_h), squeeze=False)

    for ax, (anchor, sr) in zip(axes[:, 0], items):
        task = str(sr.get("task", anchor))
        metric = str(sr.get("metric", ""))
        direction = str(sr.get("direction", "higher_better"))
        higher_is_better = direction == "higher_better"

        rows = rows_by_anchor.get(anchor, [])
        lb_rows = [rr for rr in rows if rr.get("kind") == "leaderboard"]
        ours_rows = [rr for rr in rows if rr.get("kind") == "ours"]
        lb_rows.sort(key=lambda rr: (rr.get("rank") is None, rr.get("rank") or 1e9))
        ordered = lb_rows + ours_rows

        labels: List[str] = []
        values: List[float] = []
        colors: List[str] = []
        for rr in ordered:
            kind = str(rr.get("kind"))
            model = str(rr.get("model", ""))
            rank = rr.get("rank")
            if kind == "leaderboard":
                rank_txt = "" if rank is None else f"#{int(rank)} "
                labels.append(f"{rank_txt}{model}")
            else:
                labels.append("Ours (Trimole)")
            try:
                values.append(float(rr.get("score")))
            except Exception:
                values.append(float("nan"))
            colors.append("#1f77b4" if kind == "ours" else "#9e9e9e")

        # Keep only finite values (but preserve ordering between leaderboard and ours).
        keep_idx = [i for i, v in enumerate(values) if not np.isnan(v)]
        if not keep_idx:
            ax.axis("off")
            continue

        plot_labels = [labels[i] for i in keep_idx]
        plot_values = [values[i] for i in keep_idx]
        plot_colors = [colors[i] for i in keep_idx]

        ax.barh(plot_labels, plot_values, color=plot_colors)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15, linestyle="-", linewidth=0.8)
        ax.set_xlabel(metric)

        # Title: keep it short (long strings make the tall figure unreadable).
        beat = str(sr.get("beat", ""))
        vs = str(sr.get("vs", ""))
        title_bits = [f"{task}", f"{metric}", f"top{topk}"]
        if beat:
            title_bits.append(f"beat={beat}")
        if vs:
            title_bits.append(f"vs={vs}")
        ax.set_title(" · ".join(title_bits))

        # Compact y labels a bit.
        ax.tick_params(axis="y", pad=2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


def write_topk_reports(
    ours: pd.DataFrame,
    leaderboards: Dict[str, Leaderboard],
    out_dir: Path,
    topk: int = 5,
    run_tag: str = "",
    with_html: bool = False,
    per_task: bool = False,
) -> pd.DataFrame:
    """Write per-task leaderboard top-k comparison artifacts.

    Outputs under: out_dir/run_<timestamp>/
      - all_tasks.png
      - index.html (optional, with --with-html)
      - leaderboard_topk_summary.csv
      - per_task/<task_norm>.csv/.png (optional, with --per-task; implied by --with-html)
    """
    root_name = run_tag or f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    root = out_dir / root_name
    root.mkdir(parents=True, exist_ok=True)
    per_task_effective = bool(per_task) or bool(with_html)
    per_task_dir = root / "per_task"
    if per_task_effective:
        per_task_dir.mkdir(parents=True, exist_ok=True)

    def _fmt_num(x: object) -> str:
        try:
            v = float(x)
            return "" if np.isnan(v) else f"{v:.6g}"
        except Exception:
            return str(x)

    def _fmt_signed(x: object) -> str:
        try:
            v = float(x)
            if np.isnan(v):
                return ""
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.6g}"
        except Exception:
            return ""

    def _fmt_pct(x: float) -> str:
        if np.isnan(x):
            return ""
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.2f}%"

    def _vs_text(delta: float, baseline: float) -> str:
        if np.isnan(delta):
            return ""
        if np.isnan(baseline):
            return _fmt_signed(delta)
        denom = float(abs(baseline))
        if denom < 1e-12:
            return _fmt_signed(delta)
        pct = float(delta / denom * 100.0)
        return f"{_fmt_signed(delta)} ({_fmt_pct(pct)})"

    def _beat_text(our_score: float, lb_scores: list[float], higher_is_better: bool) -> str:
        if np.isnan(our_score) or not lb_scores:
            return ""
        for i, s in enumerate(lb_scores, start=1):
            if np.isnan(s):
                continue
            better = (our_score > s) if higher_is_better else (our_score < s)
            if better:
                return f"beat {i}"
        return "-"

    summary_rows = []
    blocks_by_anchor: Dict[str, str] = {}
    rows_by_anchor: Dict[str, List[dict]] = {}
    for _, r in ours.iterrows():
        task = str(r.get("task"))
        task_norm = str(r.get("task_norm"))
        lb = leaderboards.get(task_norm)
        if lb is None:
            continue

        our_val = _our_value_for_metric(r, lb.metric_name)
        best_val = float(lb.topk.iloc[0]["Score"]) if not lb.topk.empty else float("nan")
        if np.isnan(our_val) or np.isnan(best_val):
            delta = float("nan")
        else:
            delta = (our_val - best_val) if lb.higher_is_better else (best_val - our_val)

        rows: List[dict] = []
        for _, br in lb.topk.iterrows():
            rows.append(
                {
                    "task": task,
                    "task_norm": task_norm,
                    "dataset_id": lb.dataset_id,
                    "metric": lb.metric_name,
                    "direction": "higher_better" if lb.higher_is_better else "lower_better",
                    "kind": "leaderboard",
                    "rank": int(br.get("Rank")) if not pd.isna(br.get("Rank")) else None,
                    "model": str(br.get("Model")),
                    "score": float(br.get("Score")) if not pd.isna(br.get("Score")) else float("nan"),
                }
            )
        rows.append(
            {
                "task": task,
                "task_norm": task_norm,
                "dataset_id": lb.dataset_id,
                "metric": lb.metric_name,
                "direction": "higher_better" if lb.higher_is_better else "lower_better",
                "kind": "ours",
                "rank": None,
                "model": "Trimole",
                "score": our_val,
            }
        )

        # Pre-render an inline block for the large all-in-one HTML.

        scores = [rr.get("score") for rr in rows]
        scores_num = [float(x) for x in scores if x is not None and not (isinstance(x, float) and np.isnan(x))]
        if scores_num:
            vmin = float(min(scores_num))
            vmax = float(max(scores_num))
            span = max(vmax - vmin, 1e-12)
        else:
            vmin, vmax, span = 0.0, 1.0, 1.0

        def _bar(v: object, is_ours: bool) -> str:
            try:
                x = float(v)
            except Exception:
                return ""
            if np.isnan(x):
                return ""
            width = int(min(100, max(0, (x - vmin) / span * 100)))
            color = "#1f77b4" if is_ours else "#9e9e9e"
            return (
                "<div class='scorecell'>"
                "<div class='barbg'><div class='barfg' style='width:%d%%;background:%s'></div></div>"
                "<span class='mono'>%s</span>"
                "</div>" % (width, color, _fmt_num(x))
            )

        # Build rows HTML: leaderboard first by rank, then ours.
        lb_rows = [rr for rr in rows if rr.get("kind") == "leaderboard"]
        lb_rows.sort(key=lambda rr: (rr.get("rank") is None, rr.get("rank") or 1e9))
        ours_row = [rr for rr in rows if rr.get("kind") == "ours"]
        ordered = lb_rows + ours_row

        trs = []
        for rr in ordered:
            is_ours = rr.get("kind") == "ours"
            rank = rr.get("rank")
            rank_txt = "" if rank is None else str(int(rank))
            model = str(rr.get("model", ""))
            trs.append(
                "<tr>"
                f"<td class='mono'>{rank_txt}</td>"
                f"<td>{'Ours' if is_ours else 'Leaderboard'}</td>"
                f"<td style='font-weight:{'700' if is_ours else '400'}'>{model}</td>"
                f"<td>{_bar(rr.get('score'), is_ours=is_ours)}</td>"
                "</tr>"
            )

        anchor = _safe_filename(task_norm)

        lb_scores = []
        try:
            lb_scores = [float(x) for x in lb.topk["Score"].to_list()]
        except Exception:
            lb_scores = []
        beat_txt = _beat_text(float(our_val), lb_scores, higher_is_better=lb.higher_is_better)
        vs_txt = _vs_text(float(delta), float(best_val))

        title_line = (
            f"<div class='task-title' id='{anchor}'>"
            f"<span class='task-name'>{task}</span>"
            f"<span class='task-meta'>metric: {lb.metric_name} · top{topk} · "
            f"top1: {_fmt_num(best_val)} ({str(lb.topk.iloc[0]['Model']) if not lb.topk.empty else ''}) · "
            f"ours: {_fmt_num(our_val)} · beat: <span class='mono'>{beat_txt}</span> · vs: <span class='mono'>{vs_txt}</span></span>"
            f"</div>"
        )
        block = (
            "<section class='task-block'>"
            + title_line
            + "<table class='task-table'>"
            + "<thead><tr><th>Rank</th><th>Source</th><th>Model</th><th>Score</th></tr></thead>"
            + f"<tbody>{''.join(trs)}</tbody>"
            + "</table>"
            + "</section>"
        )
        blocks_by_anchor[anchor] = block
        rows_by_anchor[anchor] = rows

        # Optional: write per-task CSV/HTML/PNG
        safe = _safe_filename(task_norm)
        out_csv = per_task_dir / f"{safe}.csv"
        out_html = per_task_dir / f"{safe}.html"
        out_png = per_task_dir / f"{safe}.png"
        plotted = False
        if per_task_effective:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            if with_html:
                _render_task_html(task=task, metric=lb.metric_name, rows=rows, out_html=out_html)
            plotted = _try_plot_task(task=task, metric=lb.metric_name, rows=rows, higher_is_better=lb.higher_is_better, out_png=out_png)

        summary_rows.append(
            {
                "task": task,
                "dataset_id": lb.dataset_id,
                "metric": lb.metric_name,
                "baseline_top1": best_val,
                "baseline_top1_model": str(lb.topk.iloc[0]["Model"]) if not lb.topk.empty else "",
                "our": our_val,
                "improvement_vs_top1": delta,
                "vs": vs_txt,
                "beat": beat_txt,
                "direction": "higher_better" if lb.higher_is_better else "lower_better",
                "topk": topk,
                "per_task_html": str(out_html.relative_to(root)) if with_html else "",
                "per_task_png": str(out_png.relative_to(root)) if per_task_effective and plotted and out_png.exists() else "",
                "anchor": anchor,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("improvement_vs_top1", ascending=False, na_position="last")

    out_summary_csv = root / "leaderboard_topk_summary.csv"
    summary.to_csv(out_summary_csv, index=False)

    # Single large PNG: all tasks in one image.
    all_png = root / "all_tasks.png"
    plotted_all = _try_plot_all_tasks_png(summary=summary, rows_by_anchor=rows_by_anchor, out_png=all_png, topk=topk)

    if with_html:
        # Single large HTML: all tasks in one scroll.
        toc_items = []
        ordered_blocks: List[str] = []
        for _, sr in summary.iterrows():
            t = str(sr.get("task", ""))
            anchor = str(sr.get("anchor") or _safe_filename(str(sr.get("dataset_id", t))))
            toc_items.append(f"<a class='toc-item' href='#{anchor}'>{t}</a>")
            b = blocks_by_anchor.get(anchor)
            if b:
                ordered_blocks.append(b)

        all_html = f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Trimole vs TDCommons Leaderboard Top{topk} (All Tasks)</title>
    <style>
        :root {{ --fg:#111; --muted:#555; --border:#eee; --bg:#fff; --chip:#f7f7f7; }}
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; color: var(--fg); background: var(--bg); }}
        .wrap {{ display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }}
        .sidebar {{ position: sticky; top: 0; height: 100vh; overflow: auto; border-right: 1px solid var(--border); padding: 18px 14px; }}
        .content {{ padding: 22px 24px; }}
        h1 {{ margin: 0 0 6px 0; font-size: 20px; }}
        .sub {{ color: var(--muted); margin: 0 0 14px 0; }}
        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
        .toc {{ display:flex; flex-direction: column; gap: 6px; margin-top: 12px; }}
        .toc-item {{ text-decoration: none; color: #1f77b4; padding: 6px 8px; border-radius: 8px; }}
        .toc-item:hover {{ background: var(--chip); }}
        .task-block {{ border: 1px solid var(--border); border-radius: 12px; padding: 14px 14px 6px 14px; margin-bottom: 14px; }}
        .task-title {{ display:flex; flex-direction: column; gap: 4px; margin-bottom: 10px; }}
        .task-name {{ font-weight: 700; font-size: 16px; }}
        .task-meta {{ color: var(--muted); font-size: 13px; }}
        .task-table {{ border-collapse: collapse; width: 100%; }}
        .task-table th, .task-table td {{ border-bottom: 1px solid var(--border); padding: 8px 6px; vertical-align: middle; font-size: 13px; }}
        .task-table th {{ text-align: left; color: var(--muted); font-weight: 600; }}
        .scorecell {{ display:flex; align-items:center; gap: 10px; }}
        .barbg {{ width: 220px; background: #f2f2f2; height: 10px; border-radius: 6px; overflow: hidden; }}
        .barfg {{ height: 10px; }}
        @media (max-width: 980px) {{
            .wrap {{ grid-template-columns: 1fr; }}
            .sidebar {{ position: relative; height: auto; border-right: none; border-bottom: 1px solid var(--border); }}
        }}
    </style>
</head>
<body>
    <div class='wrap'>
        <aside class='sidebar'>
            <h1>Top{topk} Leaderboard vs Trimole</h1>
            <div class='sub'>单页浏览全部 task。左侧是目录跳转。</div>
            <div class='sub'>CSV: <span class='mono'>{out_summary_csv.name}</span></div>
            <div class='toc'>
                {''.join(toc_items)}
            </div>
        </aside>
        <main class='content'>
            {''.join(ordered_blocks)}
        </main>
    </div>
</body>
</html>"""
        (root / "all_tasks.html").write_text(all_html, encoding="utf-8")

        # Index HTML
        rows_html = []
        for _, sr in summary.iterrows():
            t = sr.get("task", "")
            metric = sr.get("metric", "")
            base = sr.get("baseline_top1", "")
            base_m = sr.get("baseline_top1_model", "")
            ourv = sr.get("our", "")
            vs = sr.get("vs", "")
            beat = sr.get("beat", "")
            href = sr.get("per_task_html", "")
            img = sr.get("per_task_png", "")

            def fmt(x: object) -> str:
                return _fmt_num(x)

            plot_link = f"<a href='{img}'>png</a>" if isinstance(img, str) and img else ""
            rows_html.append(
                "<tr>"
                f"<td><a href='{href}'>{t}</a></td>"
                f"<td>{metric}</td>"
                f"<td>{fmt(base)}</td>"
                f"<td>{base_m}</td>"
                f"<td>{fmt(ourv)}</td>"
                f"<td style='font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace'>{beat}</td>"
                f"<td style='font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace'>{vs}</td>"
                f"<td>{plot_link}</td>"
                "</tr>"
            )

        index_html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Trimole vs TDCommons Leaderboard Top{topk}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 6px 0; }}
    .sub {{ color:#555; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 10px 8px; vertical-align: middle; }}
    th {{ text-align: left; position: sticky; top: 0; background: #fff; z-index: 1; }}
  </style>
</head>
<body>
  <h1>Trimole vs TDCommons Leaderboard Top{topk}</h1>
    <div class='sub'>推荐直接打开：<a href='all_tasks.html'>all_tasks.html</a>（单页包含所有 task）。</div>
  <div class='sub'>输出：{out_summary_csv.name}</div>
  <table>
    <thead>
      <tr>
                <th>task</th><th>metric</th><th>baseline_top1</th><th>baseline_top1_model</th><th>our</th><th>beat</th><th>vs</th><th>plot</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""
        (root / "index.html").write_text(index_html, encoding="utf-8")
    else:
        # Keep logs informative even without HTML.
        if not plotted_all:
            # No print here; caller prints.
            pass

    return summary


def _try_plot(df: pd.DataFrame, out_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    d = df.dropna(subset=["improvement"]).copy()
    if d.empty:
        return False

    d = d.head(30)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in d["improvement"].to_numpy()]

    plt.figure(figsize=(12, max(4, 0.35 * len(d) + 1)))
    plt.barh(d["task"].astype(str), d["improvement"].astype(float), color=colors)
    plt.gca().invert_yaxis()
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Trimole vs Baselines (positive = better)")
    plt.xlabel("Improvement")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    return True


def _write_html_report(df: pd.DataFrame, out_html: Path) -> None:
        out_html.parent.mkdir(parents=True, exist_ok=True)

        d = df.copy()
        # scale bars by max absolute improvement among numeric rows
        imp = pd.to_numeric(d["improvement"], errors="coerce")
        max_abs = float(np.nanmax(np.abs(imp.to_numpy()))) if np.isfinite(imp).any() else 1.0
        max_abs = max(max_abs, 1e-12)

        def bar_cell(v: object) -> str:
                try:
                        x = float(v)
                except Exception:
                        return ""
                if np.isnan(x):
                        return ""
                width = int(min(100, abs(x) / max_abs * 100))
                color = "#2ca02c" if x >= 0 else "#d62728"
                sign = "+" if x >= 0 else "-"
                return (
                        f"<div style='display:flex;align-items:center;gap:8px'>"
                        f"<div style='width:180px;background:#f2f2f2;height:10px;border-radius:6px;overflow:hidden'>"
                        f"<div style='width:{width}%;background:{color};height:10px'></div>"
                        f"</div>"
                        f"<span style='font-family:monospace'>{sign}{abs(x):.4f}</span>"
                        f"</div>"
                )

        rows_html = []
        for _, r in d.iterrows():
                rows_html.append(
                        "<tr>"
                        f"<td>{r.get('task','')}</td>"
                        f"<td>{r.get('metric','')}</td>"
                        f"<td>{'' if pd.isna(r.get('baseline_best')) else r.get('baseline_best')}</td>"
                        f"<td>{r.get('baseline_model','')}</td>"
                        f"<td>{'' if pd.isna(r.get('our')) else r.get('our')}</td>"
                        f"<td>{bar_cell(r.get('improvement'))}</td>"
                        "</tr>"
                )

        html = f"""<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Trimole vs Baselines</title>
    <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
        h1 {{ margin: 0 0 8px 0; }}
        .hint {{ color: #555; margin-bottom: 18px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border-bottom: 1px solid #eee; padding: 10px 8px; vertical-align: middle; }}
        th {{ text-align: left; position: sticky; top: 0; background: #fff; z-index: 1; }}
        td {{ font-size: 14px; }}
        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    </style>
</head>
<body>
    <h1>Trimole vs Baselines</h1>
    <div class='hint'>improvement &gt; 0 表示优于 baseline（AUROC/AUPRC: 越大越好；MAE/RMSE: 越小越好）</div>
    <table>
        <thead>
            <tr>
                <th>task</th>
                <th>metric</th>
                <th>baseline_best</th>
                <th>baseline_model</th>
                <th>our</th>
                <th>improvement</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
</body>
</html>"""

        out_html.write_text(html, encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Compare Trimole run results against TDCommons baselines")
    ap.add_argument("--baselines", type=str, default="results/baselines", help="Baselines directory")
    ap.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run directory: results/model_log/<outer_run> OR results/model_log/<outer_run>/run_YYYY...",
    )
    ap.add_argument("--out-dir", type=str, default="results/compare", help="Output directory under results")
    ap.add_argument("--topk", type=int, default=5, help="Leaderboard top-k to compare for per-task reports")
    ap.add_argument(
        "--no-topk-report",
        action="store_true",
        help="Disable per-task leaderboard top-k report artifacts (only write baseline_comparison + baseline_improvement_top30.png).",
    )
    ap.add_argument(
        "--with-html",
        action="store_true",
        help="Also write HTML reports (baseline_report.html, per-task html, index.html, all_tasks.html). Default: PNG/CSV only.",
    )
    ap.add_argument(
        "--per-task",
        action="store_true",
        help="Also write per-task artifacts under per_task/ (CSV + PNG). Implied by --with-html.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    baselines_dir = Path(args.baselines).resolve()
    run_root = Path(args.run).resolve()
    out_dir = Path(args.out_dir).resolve()

    run_dir = _find_latest_run_dir(run_root)
    baselines = load_baselines(baselines_dir)
    ours = load_ours(run_root)
    comp = compute_comparison(ours, baselines)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "baseline_comparison.csv"
    comp.to_csv(out_csv, index=False)

    out_html = out_dir / "baseline_report.html"
    if args.with_html:
        _write_html_report(comp, out_html)

    plotted = _try_plot(comp, out_dir / "baseline_improvement_top30.png")
    print(f"Wrote: {out_csv}")
    if args.with_html:
        print(f"Wrote: {out_html}")
    if plotted:
        print(f"Wrote: {out_dir / 'baseline_improvement_top30.png'}")
    else:
        print("Plot skipped (matplotlib missing or no numeric comparisons).")

    if not args.no_topk_report and args.topk and args.topk > 0:
        leaderboards = load_leaderboards(baselines_dir, topk=int(args.topk))
        summary = write_topk_reports(
            ours=ours,
            leaderboards=leaderboards,
            out_dir=out_dir,
            topk=int(args.topk),
            run_tag=run_dir.name,
            with_html=bool(args.with_html),
            per_task=bool(args.per_task),
        )
        if summary.empty:
            print("Top-k report skipped (no overlapping tasks between ours and baselines).")
        else:
            print(f"Wrote: {out_dir / run_dir.name / 'leaderboard_topk_summary.csv'}")
            all_png = out_dir / run_dir.name / "all_tasks.png"
            if all_png.exists():
                print(f"Wrote: {all_png}")
            else:
                print("Plot skipped for all_tasks.png (matplotlib/Pillow unavailable).")
            if args.with_html:
                print(f"Wrote: {out_dir / run_dir.name / 'index.html'}")
                print(f"Wrote: {out_dir / run_dir.name / 'all_tasks.html'}")


if __name__ == "__main__":
    main()
