from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path("<PROJECT_ROOT>/trimole_ept_swap_v1")
HYBRID = REPO / "results_strict" / "maplight_style_ept_hybrid_final_v1" / "summary.csv"
OUT_ROOT = REPO / "results_strict" / "strict_tdc_candidate_audit_v1"


SOURCE_RULES = {
    "frozen_ept_master_v10": {
        "status": "paper_ready_baseline",
        "risk": "low",
        "reason": "official 5-seed frozen EPT-family master; usable as baseline/result if selection was frozen before test.",
        "required_action": "None, but keep as baseline not novelty claim.",
    },
    "pgp_frozen_seedmatched_blend": {
        "status": "paper_candidate_needs_rule_doc",
        "risk": "medium",
        "reason": "5-seed seed-matched blend exists, but blend weights must be justified by train/valid/CV only.",
        "required_action": "Document weight-selection rule or rerun nested-CV weight selection before reporting.",
    },
    "repeated_tabular_fp_bagging": {
        "status": "paper_candidate_needs_5seed_formalization",
        "risk": "medium",
        "reason": "Repeated scaffold fold bagging; strong candidate, but not the official TDC 5 independent train/test protocol.",
        "required_action": "Freeze config and run official 5-seed/evaluate_many style report.",
    },
    "descriptor_sidecar_v2": {
        "status": "exploration_needs_5seed",
        "risk": "medium_high",
        "reason": "Single sidecar result; useful candidate, but not sufficient as paper result.",
        "required_action": "Freeze descriptor sidecar config from CV/valid and rerun 5 seeds.",
    },
    "chemical_prior_v2_multiblock": {
        "status": "exploration_needs_5seed_and_cv_audit",
        "risk": "medium_high",
        "reason": "Strong chemical-prior result, but current focus run is not formal 5-seed and may include broad task-wise search.",
        "required_action": "Freeze v2 variant using scaffold-CV, then rerun official 5-seed.",
    },
    "prediction_zoo_v2_valid_selected": {
        "status": "exploration_not_paper_result",
        "risk": "high",
        "reason": "Low-dimensional prediction ensemble selected on official valid, not a clean independent 5-seed model result.",
        "required_action": "Use only as candidate generator; convert selected recipe into frozen 5-seed rerun.",
    },
    "cv_selected_prediction_ensemble_v2": {
        "status": "exploration_not_paper_result",
        "risk": "high",
        "reason": "Prediction-level ensemble search; may be useful but needs nested selection and 5-seed formal audit.",
        "required_action": "Rerun with predefined candidate pool and nested CV, then report 5-seed mean/std.",
    },
    "cv_selected_prediction_ensemble_v3": {
        "status": "exploration_not_paper_result",
        "risk": "high",
        "reason": "Prediction-level ensemble search; not enough alone for strict paper result.",
        "required_action": "Use as candidate generator only.",
    },
    "tabular_fp_bagging": {
        "status": "exploration_needs_5seed",
        "risk": "medium_high",
        "reason": "Fold-bagged tabular model; not official independent 5-seed protocol.",
        "required_action": "Freeze tabular config and rerun official 5 seeds.",
    },
    "paper_main_v1_multimodal_chemical_prior": {
        "status": "exploration_needs_5seed",
        "risk": "medium_high",
        "reason": "Good systematic branch, but current output is CV-bagged single run, not official 5-seed result.",
        "required_action": "Use as branch candidate, then formal 5-seed audit.",
    },
    "layerwise_master_v4": {
        "status": "exploration_needs_formal_5seed",
        "risk": "medium_high",
        "reason": "Layer-wise selector signal is strong but not yet a fully strict 5-seed formal paper result.",
        "required_action": "Freeze selected layer via inner scaffold-CV and rerun official 5 seeds.",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(HYBRID)
    audit_rows: list[dict[str, object]] = []
    for r in rows:
        source = r["source"]
        rule = SOURCE_RULES.get(
            source,
            {
                "status": "unknown_needs_manual_audit",
                "risk": "unknown",
                "reason": "Source is not covered by strict audit rules.",
                "required_action": "Manually inspect selection and seed protocol.",
            },
        )
        strict_ready = rule["status"] == "paper_ready_baseline"
        top1_like = r.get("is_top1_level") == "True"
        audit_rows.append(
            {
                **r,
                **rule,
                "strict_paper_ready": strict_ready,
                "top1_claim_allowed_now": strict_ready and top1_like,
                "can_use_as_candidate": rule["risk"] in {"low", "medium", "medium_high"},
            }
        )

    write_csv(OUT_ROOT / "audit.csv", audit_rows)
    priority = []
    for r in audit_rows:
        if r["status"] == "paper_ready_baseline":
            continue
        score = float(r["score"])
        top1 = float(r["tdc_top1_ref"])
        gap = float(r["gap_vs_top1_ref"])
        priority.append(
            {
                "task": r["task"],
                "tdc_metric": r["tdc_metric"],
                "score": score,
                "top1_ref": top1,
                "gap_vs_top1_ref": gap,
                "source": r["source"],
                "status": r["status"],
                "risk": r["risk"],
                "required_action": r["required_action"],
                "is_top1_level_exploration": r["is_top1_level"],
            }
        )
    priority.sort(key=lambda x: (0 if x["is_top1_level_exploration"] == "True" else 1, x["gap_vs_top1_ref"]))
    write_csv(OUT_ROOT / "formal_rerun_priority.csv", priority)

    payload = {
        "n_tasks": len(audit_rows),
        "strict_paper_ready_count": sum(1 for r in audit_rows if r["strict_paper_ready"]),
        "exploration_top1_level_count": sum(1 for r in audit_rows if r["is_top1_level"] == "True"),
        "top1_claim_allowed_now_count": sum(1 for r in audit_rows if r["top1_claim_allowed_now"]),
        "priority_tasks": [r["task"] for r in priority[:8]],
    }
    (OUT_ROOT / "meta.json").write_text(json.dumps(payload, indent=2))
    (OUT_ROOT / "README.md").write_text(
        "# Strict TDC Candidate Audit v1\n\n"
        "This audit separates defense/exploration scores from results that can be reported under a strict TDC paper protocol. "
        "Only configurations with frozen train/valid selection and official independent 5-seed evaluation should be used as paper claims.\n"
    )
    print(json.dumps(payload, indent=2))
    print(OUT_ROOT / "audit.csv")
    print(OUT_ROOT / "formal_rerun_priority.csv")


if __name__ == "__main__":
    main()
