"""Apply the four recorded final combinations without fitting or selection."""
import numpy as np
from scipy.stats import rankdata


def vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(name + " must be a nonempty finite prediction vector.")
    return value


def aligned(parts):
    if len({len(p) for p in parts}) != 1:
        raise ValueError("Component predictions must have identical row counts and row order.")


def rank_n(value):
    return rankdata(value, method="average") / len(value)


def zscore_seed(value):
    # Matches build_cyp2d6_ap_xl_descriptor_seedbag_endpoint_v1.py.
    value = np.asarray(value, dtype=np.float64)
    std = float(np.std(value))
    return np.zeros_like(value) if std == 0 else (value - float(np.mean(value))) / std


def seed_vectors(predictions, rule, run_index):
    groups = rule["formal_seed_groups"]
    if not 1 <= run_index <= len(groups):
        raise ValueError("--run-index must be 1 through 5.")
    seeds = groups[run_index - 1]
    if "seedbag_heads" in predictions:
        values = np.asarray(predictions["seedbag_heads"], dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(seeds):
            raise ValueError("seedbag_heads must contain the five raw seed vectors for one formal run, shape (5, n_rows).")
        if "seed_ids" in predictions:
            ids = np.asarray(predictions["seed_ids"])
            if ids.shape != (len(seeds),) or not np.array_equal(ids, np.asarray(seeds)):
                raise ValueError("seed_ids do not match the recorded formal run.")
        parts = [vector(row, "seedbag_heads") for row in values]
    else:
        names = ["seed_" + str(seed) for seed in seeds]
        missing = [name for name in names if name not in predictions]
        if missing:
            raise ValueError("Missing raw predictions for the recorded seed group: " + ", ".join(missing))
        parts = [vector(predictions[name], name) for name in names]
    aligned(parts)
    return parts


def seedbag_mean(predictions, rule, run_index):
    if "seedbag_mean" in predictions:
        return vector(predictions["seedbag_mean"], "seedbag_mean")
    return np.mean(np.stack(seed_vectors(predictions, rule, run_index), axis=0), axis=0)


def hepatic_prediction(predictions, rule):
    names = rule["base_members"]
    if all(name in predictions for name in names):
        base_parts = [vector(predictions[name], name) for name in names]
        aligned(base_parts)
        base = np.mean([rank_n(part) for part in base_parts], axis=0)
    elif "frozen_feature_heads" in predictions:
        raw = np.asarray(predictions["frozen_feature_heads"], dtype=np.float64)
        if raw.ndim == 1:
            # Already-completed frozen-head rank mean from the archived component.
            base = vector(raw, "frozen_feature_heads")
        elif raw.ndim == 2 and raw.shape[0] == len(names):
            base_parts = [vector(row, "frozen_feature_heads") for row in raw]
            aligned(base_parts)
            base = np.mean([rank_n(part) for part in base_parts], axis=0)
        else:
            raise ValueError("frozen_feature_heads must be a completed vector or the three recorded member vectors.")
    else:
        raise ValueError("Supply c060, c070 and c008, or the completed frozen_feature_heads component.")
    fine = vector(predictions["kpgt_finetuned"], "kpgt_finetuned")
    staged = vector(predictions["kpgt_staged"], "kpgt_staged")
    unimol = vector(predictions["unimol"], "unimol")
    aligned([base, fine, staged, unimol])
    weight = rule["prior_fine_tune_weight"]
    prior = (1 - weight) * rank_n(base) + weight * rank_n(fine)
    weight = rule["staged_weight"]
    staged_fused = (1 - weight) * rank_n(prior) + weight * rank_n(staged)
    weight = rule["unimol_weight"]
    return (1 - weight) * rank_n(staged_fused) + weight * rank_n(unimol)


def fixed_prediction(task, predictions, rule, run_index=1):
    if not 1 <= run_index <= 5:
        raise ValueError("--run-index must be 1 through 5.")
    operation = rule["operation"]
    if operation == "nested_rank_blend":
        return hepatic_prediction(predictions, rule)
    if operation == "mean_of_five_raw_predictions":
        return seedbag_mean(predictions, rule, run_index)
    if operation == "raw_seedbag_fp_blend":
        heads = seedbag_mean(predictions, rule, run_index)
        fp = vector(predictions["fp_only"], "fp_only")
        aligned([heads, fp])
        return rule["seedbag_weight"] * heads + rule["fp_only_weight"] * fp
    if operation == "mean_of_seedwise_zscore_blends":
        heads = seed_vectors(predictions, rule, run_index)
        xl = vector(predictions["xl"], "xl")
        descriptor = vector(predictions["descriptor_sidecar_v2"], "descriptor_sidecar_v2")
        aligned(heads + [xl, descriptor])
        weights = rule["weights"]
        # Normalize each AP seed before averaging, as in the archived builder.
        parts = [weights[0] * zscore_seed(ap) + weights[1] * zscore_seed(xl)
                 + weights[2] * zscore_seed(descriptor) for ap in heads]
        return np.mean(parts, axis=0)
    raise ValueError("Unknown fixed combination for " + task)
