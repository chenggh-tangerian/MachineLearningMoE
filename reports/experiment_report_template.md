# 实验记录模板

## 1. 实验名称

- 例如：K-Means 路由 MoE 的负载均衡与可行性实验

## 2. 实验目标

- 验证 K-Means 路由是否改善专家负载均衡。
- 验证 K-Means 路由相对随机路由的稳定性收益。
- 验证 MoE 专家层相对普通 Linear 基线的效果。

## 3. 实验配置

- 数据集：
- 模型配置：
- 专家数：
- top-k：
- 设备：
- 种子：

## 4. 评测指标

- 分类准确率：
- 训练损失：
- 路由熵：
- 负载均衡度：
- 负载变异系数：
- 吞吐（samples/s）：
- 吞吐对比：

## 5. 结果表

| 方案 | Accuracy | Loss | Routing Entropy | Load Balance | Load CV | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| KMeans MoE |  |  |  |  |  |  |
| Random Routing Baseline |  |  |  |  |  |  |
| Linear Expert Baseline |  |  |  |  |  |  |

## 6. 结论

- 本次实验说明了什么：
- 主要瓶颈：
- 下一步优化：
