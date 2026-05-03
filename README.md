# KMeans-MoE

一个围绕 MoE、K-Means 路由和路由可行性验证构建的课程项目骨架。

这个版本专注算法本身，不再强调系统加速部分。重点是回答一个问题：用 K-Means 作为路由机制，是否真的能让 MoE 更稳定、更均衡、更容易解释。

## 项目目标

1. 使用 K-Means 作为传统机器学习路由器，降低 MoE 的专家塌缩和路由不稳定问题。
2. 提供随机路由和 Linear 专家两类基线，形成完整的消融对比。
3. 提供从数据、路由、专家计算到训练脚本的完整项目结构，方便写报告和做实验。

## 目录结构

```text
.
├── scripts/
│   ├── benchmark.py
│   ├── compare_baselines.py
│   ├── evaluate.py
│   └── train.py
├── src/
│   └── moe_project/
│       ├── baselines.py
│       ├── config.py
│       ├── data.py
│       ├── experts.py
│       ├── model.py
│       └── router.py
├── requirements.txt
├── reports/
│   ├── baseline_comparison_template.csv
│   ├── benchmark_results_template.csv
│   └── experiment_report_template.md
└── README.md
```

## 运行方式

先安装依赖：

```bash
python -m pip install -r requirements.txt
```

然后运行一个 CPU 可用的烟雾测试：

```bash
PYTHONPATH=src python scripts/train.py --epochs 5 --device cpu --dataset digits
```

如果想把训练过程保存成结果文件，可以这样运行：

```bash
PYTHONPATH=src python scripts/train.py --epochs 20 --device cpu --dataset digits --save-dir outputs/train --save-checkpoint
```

训练结束后会生成 `outputs/train/history.json` 和可选的 `outputs/train/model.pt`。

评测与基准测试脚本也已经补齐：

```bash
PYTHONPATH=src python scripts/evaluate.py --device cpu --dataset digits --output-json outputs/eval/summary.json
PYTHONPATH=src python scripts/benchmark.py --device cpu --dataset digits --output-csv reports/benchmark_results.csv
PYTHONPATH=src python scripts/compare_baselines.py --device cpu --dataset digits --output-dir outputs/baselines
```

## 项目里怎么理解 MoE

当前工程把专家前向拆成“按专家分组的 token chunk”，然后将它们交给专家网络处理，再按路由权重加权合并。你可以把它看成一个可解释的稀疏专家系统：

1. K-Means 路由减少 token 到专家的搜索空间。
2. top-k 路由让每个 token 只和少量专家交互。
3. 加权聚合保证输出仍然连续可训练。

## 建议的实验

1. 先在合成分类数据上验证 K-Means 路由是否比随机路由更稳定。
2. 再切到 Digits 标准数据集，验证方法是否能在更正式的基准上稳定工作。
3. 统计负载均衡指标、路由熵、top-k 命中率。
4. 对比普通 `Linear` 专家和 MoE 版本在吞吐上的差异。
5. 用 `compare_baselines.py` 同时跑 K-Means 路由、随机路由和 Linear 专家消融，形成一张统一对比表。

## 结果记录

实验记录模板位于 [reports/experiment_report_template.md](reports/experiment_report_template.md)，吞吐结果模板位于 [reports/benchmark_results_template.csv](reports/benchmark_results_template.csv)。基线对比表模板位于 [reports/baseline_comparison_template.csv](reports/baseline_comparison_template.csv)。

建议每次实验都记录以下字段：

1. 模型版本与参数。
2. 数据集规模与随机种子。
3. 训练损失、准确率、路由熵、负载均衡度、负载变异系数。
4. 吞吐、显存或内存占用。
