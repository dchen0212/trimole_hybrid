from pathlib import Path
from tdc.benchmark_group import admet_group

OUT = Path("data/data_benchmark_tdc_official")
OUT.mkdir(parents=True, exist_ok=True)

group = admet_group(path=str(OUT.parent))

for name in group.dataset_names:
    bench = group.get(name)
    train, valid = group.get_train_valid_split(benchmark=bench["name"], split_type="default", seed=42)
    test = bench["test"]

    task_dir = OUT / name
    task_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(task_dir / "train.csv", index=False)
    valid.to_csv(task_dir / "valid.csv", index=False)
    test.to_csv(task_dir / "test.csv", index=False)

    print(f"[done] {name}: train={len(train)} valid={len(valid)} test={len(test)}")

print("\nSaved to:", OUT)
