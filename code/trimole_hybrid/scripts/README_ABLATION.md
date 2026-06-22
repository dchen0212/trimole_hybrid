# 单模态消融实验使用指南

## 概述

这个消融实验会分别训练三个单模态版本的模型，帮助你识别哪个模态对模型性能的贡献最大：

- **ChemBERTa (SMILES)**：只使用分子的SMILES序列表示
- **KPGT (Graph)**：只使用分子的图结构表示  
- **Uni-Mol (3D)**：只使用分子的3D构象表示

## 快速开始

### 运行完整消融实验

```bash
cd <PROJECT_ROOT>/trimole

# 一键运行所有三个单模态实验
bash scripts/ablation_single_modality.sh
```

这个脚本会：
1. 依次训练三个单模态模型（ChemBERTa、KPGT、Uni-Mol）
2. 自动生成对比分析报告和可视化
3. 保存所有结果到 `results/ablation_single_modality/`

### 后台运行（推荐）

因为实验需要较长时间，建议后台运行：

```bash
cd <PROJECT_ROOT>/trimole
nohup bash scripts/ablation_single_modality.sh > ablation.log 2>&1 &

# 查看运行状态
tail -f ablation.log
```

## 输出结果

运行完成后，结果将保存在 `results/ablation_single_modality/` 目录下：

```
results/ablation_single_modality/
├── run_20260127_1234_chemberta_only/      # ChemBERTa单模态结果
│   ├── results_all.csv
│   └── <task_name>/
│       ├── best_model.pth
│       ├── meta.json
│       └── history.json
├── run_20260127_1234_kpgt_only/           # KPGT单模态结果  
│   └── ...
├── run_20260127_1234_unimol_only/         # Uni-Mol单模态结果
│   └── ...
└── comparison_20260127_1234/              # 对比分析结果
    ├── ablation_summary.csv               # 详细对比表格
    ├── ablation_stats.json                # 汇总统计
    ├── ablation_comparison.png            # 可视化对比图
    ├── tasks_best_on_chemberta.csv        # ChemBERTa最优任务
    ├── tasks_best_on_kpgt.csv             # KPGT最优任务
    └── tasks_best_on_unimol.csv           # Uni-Mol最优任务
```

### 关键文件说明

1. **ablation_summary.csv**：完整对比表格，包含：
   - 每个任务在三个模态上的表现
   - 与TDCommons baseline的对比
   - 最佳模态标记

2. **ablation_comparison.png**：可视化图表，包含：
   - 绝对性能对比（与baseline对比）
   - 相对性能对比（百分比）

3. **ablation_stats.json**：汇总统计，包含：
   - 各模态"最佳"的任务数量
   - 平均改进幅度
   - 胜率（超过baseline的任务比例）

## 手动运行单个模态

如果只想测试某一个模态，可以手动运行：

```bash
cd <PROJECT_ROOT>/trimole

export UNIMOL_WEIGHT_DIR=<PROJECT_ROOT>/trimole/data/weights
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=<ENV_ROOT>/kpgt/lib:$LD_LIBRARY_PATH

# 只用ChemBERTa (SMILES)
<ENV_ROOT>/trimole/bin/python -m trimole.pipelines.batch_run_data_new \
  --data-new ./data/data_new \
  --out ./results/test_chemberta \
  --modalities chemberta \
  --use-task-configs \
  --task-config-variant adaptive \
  --baselines-dir ./results/baselines

# 只用KPGT (Graph)
<ENV_ROOT>/trimole/bin/python -m trimole.pipelines.batch_run_data_new \
  --data-new ./data/data_new \
  --out ./results/test_kpgt \
  --modalities kpgt \
  --use-task-configs \
  --task-config-variant adaptive \
  --baselines-dir ./results/baselines

# 只用Uni-Mol (3D)
<ENV_ROOT>/trimole/bin/python -m trimole.pipelines.batch_run_data_new \
  --data-new ./data/data_new \
  --out ./results/test_unimol \
  --modalities unimol \
  --use-task-configs \
  --task-config-variant adaptive \
  --baselines-dir ./results/baselines
```

## 只跑部分任务（快速测试）

如果想先在几个任务上测试，可以使用 `--tasks` 参数：

```bash
# 只跑3个任务做快速测试
bash scripts/ablation_single_modality.sh --tasks ames bioavailability_ma bbb_martins
```

或者修改 `ablation_single_modality.sh` 脚本，在 `COMMON_ARGS` 中添加：

```bash
COMMON_ARGS="
  ...
  --tasks ames bioavailability_ma bbb_martins
"
```

## 自定义超参数

如果想调整训练参数，修改 `ablation_single_modality.sh` 中的 `COMMON_ARGS`：

```bash
COMMON_ARGS="
  --data-new $PROJECT_ROOT/data/data_new
  --max-epochs 100           # 调整最大epoch
  --patience 20              # 调整早停patience
  --batch-size 128           # 调整batch size
  --lr 3e-4                  # 调整学习率
  --hidden-dim 256           # 调整隐藏层维度
  --weight-decay 0.02        # 调整权重衰减
  --use-task-configs         # 使用任务特定配置
  --task-config-variant adaptive
  --baselines-dir $PROJECT_ROOT/results/baselines
"
```

## 分析现有结果

如果已经有三个单模态的运行结果，可以直接运行分析脚本：

```bash
<ENV_ROOT>/trimole/bin/python scripts/analyze_ablation_results.py \
  --chemberta-run ./results/path/to/chemberta_run \
  --kpgt-run ./results/path/to/kpgt_run \
  --unimol-run ./results/path/to/unimol_run \
  --baselines-dir ./results/baselines \
  --out-dir ./results/my_comparison \
  --top-k 30
```

## 预期运行时间

根据数据集大小和硬件配置：
- 每个模态：约 2-4 小时（所有22个任务）
- 总计：约 6-12 小时
- 分析脚本：< 1 分钟

## 解读结果

### 1. 查看汇总统计

```bash
cat results/ablation_single_modality/comparison_*/ablation_stats.json
```

关注：
- `best_modality_counts`：哪个模态最优任务最多
- `avg_improvement_vs_baseline`：平均改进幅度
- `win_rate_vs_baseline`：胜率

### 2. 查看详细表格

```bash
# 按最佳模态排序
cat results/ablation_single_modality/comparison_*/ablation_summary.csv | \
  grep -v "^task," | \
  sort -t, -k11,11
```

### 3. 查看可视化

打开 `ablation_comparison.png` 查看：
- 上图：绝对性能对比
- 下图：相对于baseline的百分比

### 典型发现示例

- **ChemBERTa (SMILES)** 通常在以下任务表现好：
  - 需要化学结构识别的任务
  - 训练集较小的任务

- **KPGT (Graph)** 通常在以下任务表现好：
  - 需要拓扑结构信息的任务
  - 药物代谢/相互作用任务

- **Uni-Mol (3D)** 通常在以下任务表现好：
  - 需要空间构象的任务（如BBB穿透）
  - 需要立体化学的任务

## 常见问题

### Q: 某个模态运行失败怎么办？

检查：
1. embedding文件是否存在：`data/data_new/<task>/embeddings/*.npy`
2. 查看错误日志：`cat ablation.log`
3. 单独重跑失败的模态

### Q: 如何对比单模态和三模态融合？

运行一次三模态实验（`--modalities all`），然后手动对比：

```python
import pandas as pd

# 加载结果
trimodal = pd.read_csv("results/model_log/run_*/results_all.csv")
chemberta = pd.read_csv("results/ablation_single_modality/run_*_chemberta_only/results_all.csv")

# 合并对比
merged = trimodal.merge(chemberta, on="task", suffixes=("_tri", "_cb"))
merged["improvement"] = merged["primary_metric_tri"] - merged["primary_metric_cb"]
print(merged[["task", "primary_metric_tri", "primary_metric_cb", "improvement"]])
```

### Q: 内存不足怎么办？

减小batch size：

```bash
# 修改脚本中的 --batch-size 参数
--batch-size 32  # 原来是64
```

## 进阶：与主README中的全模态对比

消融实验完成后，可以与 `README.md` 中记录的全模态实验对比，生成一个"模态贡献分析"：

```python
# 示例：计算模态贡献
# fusion_score - best_single_modality = 模态协同增益
# best_single_modality - worst_single_modality = 模态差异
```

这可以帮助你回答：
- 哪个模态是"必需"的？
- 哪个模态可能是"锦上添花"？
- 模态融合的协同增益有多大？
