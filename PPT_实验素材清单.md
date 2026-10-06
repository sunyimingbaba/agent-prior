# CUP 改造项目：PPT 实验素材清单

## 一、项目主线

原始 CUP/CCB 是基线；本项目的核心改造是用 MLLM Agent 产生的视觉语义先验替换/增强 CCB 先验，并进一步加入领域工具、SFT/GRPO 训练和奖励设计。

主线结论：Agent prior 在多个数据集和噪声条件下提升检索性能，尤其在 RSITMD 和困难查询上明显；工具和奖励需要谨慎设计，否则会出现“形式奖励高、工具实际不调用”的 reward hacking。

## 二、建议 PPT 的实验顺序

### 1. 总览：改造是否有效

图：[subexperiment_summary.png](/Users/botlv/CUP-code/figs/subexperiment_summary.png)

包含：UCM/RSICD/RSITMD 训练曲线、奖励设计、困难度分层、数据效率。适合作为实验总览页。

注意：上排曲线是 validation mR，不要与最终 test 表中的 mR 混用。

### 2. 主结果：CCB vs Agent prior

可在表格页放以下最终 test mR：

| 方法 | UCM | RSICD | RSITMD |
|---|---:|---:|---:|
| Original CUP / CCB | 50.85 | 31.15 | 40.25 |
| Agent prior（纯推理，无工具/无 RL） | 59.49 | 45.69 | 68.17 |
| Agent + domain tool + GRPO（当前主方法） | 58.55 | 43.12 | 58.63 |

### 3. 训练策略：SFT、冷启动 GRPO 与收敛

图：[main_curves.png](/Users/botlv/CUP-code/figs/main_curves.png)

说明：比较 Original CUP/CCB、SFT-init、cold-start GRPO；用于解释训练策略和收敛过程。

补充图：[convergence_compare.png](/Users/botlv/CUP-code/figs/convergence_compare.png)

RSITMD 上 Agent prior 早期达到较高性能，收敛速度快于 CCB。

### 4. 工具与先验变体消融

建议在表格中列出：

- Agent 纯推理、无工具/无 RL：UCM 59.49，RSICD 45.69，RSITMD 68.17。
- naive tool：57.23 / 44.52 / 63.26。
- random tool：56.47 / 42.75 / 58.16。
- domain tool + Agent + GRPO：58.55 / 43.12 / 58.63。
- 旧 zero-shot tool + Agent + GRPO：54.90 / 39.00 / 59.78。

图：[val_curves.png](/Users/botlv/CUP-code/figs/val_curves.png)

### 5. 奖励设计：性能提升与 reward hacking

图：[subexperiment_summary.png](/Users/botlv/CUP-code/figs/subexperiment_summary.png)（左下子图）

关键结果：format-only 的 mR 较高，但工具注入率为 0%，说明模型可能只学会满足格式，而没有真正调用工具；加入硬排序奖励或不同 lambda 后，注入率和性能发生明显变化。

可放表：format-only 63.66/0%，hard rank 61.64/0%，lambda=0.5 为 57.94/98.9%，lambda=1.0 为 57.97/90.3%，lambda=2.0 为 56.97/75.4%，pure baseline 62.05。这里的百分比是工具注入率。

### 6. 困难查询分层：Agent 主要帮助哪里

图：[hard_strata.png](/Users/botlv/CUP-code/figs/hard_strata.png)

UCM T2I R@1：Agent 相对 CCB 在 hard/medium/easy 上分别约提升 +4.3/+5.7/+0.0 个百分点。

补充图：[hard_strata_rsitmd.png](/Users/botlv/CUP-code/figs/hard_strata_rsitmd.png)

RSITMD 中 Agent 相对 CLIP zero-shot 在 hard/medium/easy 上提升约 +4.7/+26.5/+53.0 个百分点；这是 Agent vs CLIP，不是 Agent vs CCB，图注要写清楚。

### 7. 数据效率

图：[data_scale.png](/Users/botlv/CUP-code/figs/data_scale.png)

UCM Agent prior 使用 25%/50%/100% 数据时 mR 约为 45.9/49.6/58.5；50% 数据的 Agent 已接近 CCB 全数据基线 50.85。

### 8. 噪声鲁棒性

主图：[noise_robustness_summary.png](/Users/botlv/CUP-code/figs/noise_robustness_summary.png)

辅助原图：[noise_robust.png](/Users/botlv/CUP-code/figs/noise_robust.png)、[noise_robust_rsitmd.png](/Users/botlv/CUP-code/figs/noise_robust_rsitmd.png)

当前已测噪声比例为 0%/20%/40% word-drop，不要在 PPT 中写成 60%/80%。UCM 同时有 Agent/CCB/CLIP；RSITMD 当前图是 Agent vs CLIP。

### 9. 同一 query 下的定性 Top-5 对照

首选图：[qualitative_ccb_vs_agent.png](/Users/botlv/CUP-code/figs/qualitative_ccb_vs_agent.png)

含义：同一个文本 query，左侧是 CCB Top-5，右侧是 Agent Top-5，绿色框是真实匹配图。示例中真实图由 CCB 第 98/52/43 名提升到 Agent 第 4/3/4 名。这页最适合解释“Agent 到底改进了什么”。

旧图：[qualitative_cases.png](/Users/botlv/CUP-code/figs/qualitative_cases.png) 只有 Ours 的成功/失败案例，不建议作为主要对照图。

### 10. 特征可视化（补充页）

图：[tsne_compare.png](/Users/botlv/CUP-code/figs/tsne_compare.png)

CCB 的 silhouette 约 0.698，高于 Agent 约 0.659；因此不要把它宣传为“Agent 特征分离优于 CCB”。可以作为诊断页，说明检索提升不一定等同于全局聚类指标提升。

## 三、可直接用于 PPT 的页面结构

1. 研究问题与改造：CCB prior → Agent prior。
2. 实验设置：UCM、RSICD、RSITMD；I2T/T2I；R@1/R@5/R@10 与 mR。
3. 主结果表：Original CUP/CCB、Agent prior、完整方法。
4. 训练策略与收敛：`main_curves.png`。
5. 工具与奖励消融：工具变体 + reward hacking。
6. 困难查询分层：`hard_strata.png`。
7. 数据效率与噪声鲁棒性：`data_scale.png` + `noise_robustness_summary.png`。
8. 同 query Top-5 定性对照：`qualitative_ccb_vs_agent.png`。
9. 外部论文对比表：DFIM、PDAR-RSITR、RRSITR（单独一页，注明 backbone、数据集和噪声定义不同）。

## 四、汇报时必须注明的口径

- validation 曲线不能直接当作最终 test 结果。
- 噪声实验目前只有 0%/20%/40% word-drop。
- RSITMD 难度图是 Agent vs CLIP，不是 Agent vs CCB。
- RRSITR 的噪声是训练图文错配噪声，与本项目的 query word-drop 不是同一种噪声。
- reward 实验中高分但 0% 工具注入是 reward hacking 证据，不应作为完整方法的性能结论。
