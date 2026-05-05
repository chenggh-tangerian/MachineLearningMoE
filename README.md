# KMeans-MoE 课程作业提交版

本项目验证传统机器学习路由（K-Means、MRF、Random）在 MoE 中的可行性，并给出统一基线对比。

当前提交版聚焦两个任务：
1. CIFAR-100 图像分类（Accuracy）
2. WikiText2 语言建模（PPL）

## 核心内容

1. MoE 分类模型与路由基线：K-Means / MRF / Random / Linear experts
2. WikiText2 语言模型版本的 MoE 路由对比
3. 一键生成作业表格脚本（对应最终对比表）

## 目录说明

```text
.
├── run/
│   └── run_routing_table.py            # 一键跑 CIFAR-100 + WikiText2 并汇总表格
├── scripts/
│   ├── compare_baselines.py            # 分类基线对比（含 CIFAR-100）
│   ├── run_wikitext2_experiment.py     # WikiText2 路由对比
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
├── src/moe_project/
│   ├── data.py                         # 数据加载（含 CIFAR-100、WikiText2）
│   ├── model.py                        # 分类 MoE
│   ├── language_model.py               # 语言模型 MoE
│   ├── router.py
│   ├── experts.py
│   ├── baselines.py
│   ├── metrics.py
│   └── config.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 环境安装

```bash
python -m pip install -r requirements.txt
```

## 一键复现实验（最终结果）

```bash
PYTHONPATH=src python run/run_routing_table.py \
	--device cpu \
	--seed 42 \
	--output-dir outputs/routing_table_final
```

说明：
1. 不要使用 `--smoke`，那是连通性检查模式，不用于最终汇报。
2. 首次运行会自动下载 CIFAR-100 和 WikiText2 数据。

## 输出文件

运行完成后会在输出目录得到：
1. `routing_table.csv`：最终可交表格数据
2. `routing_table.md`：最终可直接粘贴到报告的表格
3. `cifar100/baseline_comparison.csv`：分类任务明细
4. `wikitext2/seed_42/baseline_comparison.json`：语言任务明细

## 单独运行（可选）


```bash
PYTHONPATH=src python scripts/compare_baselines.py --dataset cifar100 --device cpu --output-dir outputs/cifar100_only
PYTHONPATH=src python scripts/run_wikitext2_experiment.py --device cpu --output-dir outputs/wikitext2_only
```

## 本实验启动命令

PYTHONPATH=src python run/run_routing_table.py --device cpu --seed 42 --output-dir outputs/routing_table_final
