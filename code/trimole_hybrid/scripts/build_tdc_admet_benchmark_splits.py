from pathlib import Path
import pandas as pd

from tdc.benchmark_group import admet_group

OUT_ROOT = Path("data/data_benchmark")
SEED = 42  # 用固定 seed 切 train/valid；test 由 benchmark 固定给出

OUT_ROOT.mkdir(parents=True, exist_ok=True)

group = admet_group(path=str(OUT_ROOT.parent.resolve()))
names = list(group.dataset_names)

print("ADMET benchmark tasks:", names)
print("Total:", len(names))

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    colmap = {}
    if "Drug" in df.columns:
        colmap["Drug"] = "smiles"
    if "Y" in df.columns:
        colmap["Y"] = "label"
    if "SMILES" in df.columns:
        colmap["SMILES"] = "smiles"
    df = df.rename(columns=colmap)

    preferred = []
    for c in ["Drug_ID", "smiles", "label"]:
        if c in df.columns:
            preferred.append(c)

    other = [c for c in df.columns if c not in preferred]
    df = df[preferred + other]
    return df

summary = []

for name in names:
    print(f"\n=== building {name} ===")
    bench = group.get(name)
    train, valid = group.get_train_valid_split(benchmark=name, split_type="default", seed=SEED)
    test = bench["test"]

    task_dir = OUT_ROOT / name
    task_dir.mkdir(parents=True, exist_ok=True)

    train = normalize_cols(train)
    valid = normalize_cols(valid)
    test = normalize_cols(test)

    train.to_csv(task_dir / "train.csv", index=False)
    valid.to_csv(task_dir / "valid.csv", index=False)
    test.to_csv(task_dir / "test.csv", index=False)

    summary.append({
        "task": name,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_total": len(train) + len(valid) + len(test),
    })

    print(f"{name}: train={len(train)}, valid={len(valid)}, test={len(test)}, total={len(train)+len(valid)+len(test)}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_ROOT / "_benchmark_split_summary.csv", index=False)

print("\nSaved summary:", OUT_ROOT / "_benchmark_split_summary.csv")
print(summary_df.to_string(index=False))
