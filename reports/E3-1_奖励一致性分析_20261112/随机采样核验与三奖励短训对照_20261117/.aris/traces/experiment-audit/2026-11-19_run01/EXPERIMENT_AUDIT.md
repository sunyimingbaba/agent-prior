# Experiment Audit Report

日期：2026-11-19  
项目：UCM 数量诊断下的随机采样与三奖励继续短训对照  
复核者：独立新 Codex 子代理 `/root/final_comparison_audit`；具体模型版本和 reasoning 配置 [待确认]。  
审查独立性：`same-family`；语义结论采纳状态：`provisional`。  
总体判断：**WARN**；完整性状态：**warn**；实验执行完成，预设采纳判定为 **false**。

## 证据位置与复核范围

原始根目录：`/Users/botlv/CUP-code/reports/E3-1_奖励一致性分析_20261112/随机采样核验与三奖励短训对照_20261117/统一评估_20261119/原始运行记录`。下文 `code/...`、`inputs/...`、`analysis.json` 等引用均相对此目录。独立检查脚本、运行日志、机器复核结果保存在本报告同目录：

- `fresh_reviewer_checks.py`
- `fresh_reviewer_checks.log`
- `fresh_reviewer_deterministic_checks.json`

本次读取全部指定评估、分析、训练和恢复脚本，核对三组训练 result/updates/samples、四策略 records/retrieval/result、锁定协议和最终完成记录；另读 rollout、概率重放和独立核验脚本。独立脚本直接读取本机原始 Arrow、用户人工确认台账、保存张量和最终 optimizer/adapter。全过程为只读核查；写入仅限本次复核目录。

## A. 真值来源：PASS

数量真值取自 `inputs/quantity_cases.json` 的 `actual_count`，包含人工确认来源、图像 RGB 哈希及原始描述；`reward_model_v2.py:64–69` 以该数量确定正确列和其他数量列。85 个数量均已与 `reports/E1_UCM试验_20261110/第九步_去重与补充/100组工作清单.csv` 逐条匹配，原目标名称也匹配。沙坑名称的统一保留于 `original_object`。A/B 文本是依据真值构造的固定模板，标签来源为人工确认。

最终检索图库由原始 UCM val Arrow 的 210 图、每图 5 描述构成，见 `evaluate_policy_v2.py:41–52`。本机重算 train/val Arrow SHA256，与 `inputs/dataset_sources_reference.json` 相同；273 个 cohort 图像的行号、文件名、RGB 哈希全部核对通过，图库 1050 条描述及图像归属顺序全部匹配。

`repository_duplicate_token_positive` 把 tokenize 后相同的描述视为共享正例，`strict_index` 使用原图归属，`strict_tie_averaged` 在 strict 正例定义上平均同分排序。见 `evaluate_policy_v2.py:54–61` 和 `matched_common_v3.py:124–146`。当前图库有 391 个 token group，三种协议必须保留名称。

工具反馈来自相同图像既有真实分类结果缓存，并在模型发起 scene_classify 后接入；见 `stochastic_rollout.py:31–33` 与 cohort 的反馈来源字段。该反馈作为模型输入。proxy reward 的目标为训练图库中的图像身份，其与真实语义检索的关系需按代理奖励叙述。

## B. 归一化：PASS

数量分数是单位化图像/文本特征的点积，pair/hard margin 分别为正确列减固定错误列/同对象其他缓存数量最大值；准确率按 margin > 0 计算，同分记错误。proxy reward 是 `1/(1+严格更高分候选数)`；完整协议奖励为 0.2。定义见 `reward_model_v2.py:56–69`。评估准确率和 mR 采用样本比例及固定六指标平均。

训练 advantage 使用组内 8 个 reward 的样本标准差，见 `train_reward_arm_v2.py:38–46`；该变换属于训练梯度计算，报告仍保留原始 reward 和 score。归档独立 CUDA 核查验证原始 mean/std/advantage，CPU 核查单独处理 float32 reduction 差异，见 `normalization_verification_addendum.json` 与 `verify_training_cpu_v2.py:15–28`。本次重新核对各组 informative 判定及实际更新跳过事件；proxy_rank 第 12 次尝试为零方差并跳过 optimizer。

## C. 结果存在性与真实性：PASS

本机直接核验：

- 4 策略 × 383 = **1532 条最终评估**，全部 tensor/trace 文件哈希、图像身份和顺序匹配；全部 trace 协议标志重新分类。
- **780 条带数量标签的评估张量**重建 A/B pair/hard margin 与正确性，单位图像特征点积误差 < 1e-6。
- 3 组 × 256 = **768 条训练样本**的 tensor/trace 哈希通过，96 次尝试的样本顺序、种子、策略哈希链与更新计数匹配。
- 3 个最终 adapter 各 **584 个 FP32 LoRA 张量**重建策略哈希，最终 optimizer 的 584 个 state.step 分别为 **31、32、32**。
- quantity_pair 暂停前 **96 条样本/12 次尝试**的字节前缀哈希与 `resume_20261119/restore.json` 一致；恢复实现逐项核查 adapter、optimizer 与四类 RNG，见 `restore_training_group.py:41–100`。当前脚本从第 13 次尝试继续，见 `train_reward_arm_resume_v3.py:30–38`。
- `post_job_result_v3.json` 为 complete，其中 analysis、raw/training/metric verification、final assets 共 5 个哈希全部重算匹配。
- `continuation_job_result_v3.json` 的训练和评估完成情况与逐策略记录一致。最终检查点为固定第 32 次尝试，见 `train_reward_arm_v2.py:49–51`。

归档独立原始核验另覆盖 2436 条记录、4863 个生成轮次；其中包含 pilot 和早期训练记录，故与本次 1532 条最终评估范围不同。本次没有重新执行 GPU 生成、梯度更新或完整神经网络前向。hidden state 到读出器特征的执行链依靠原始脚本、冻结检查和归档执行证据；本机直接重算覆盖后续点积、指标、区间和检查点状态。

## D. 评估实现覆盖：PASS

`evaluate_policy_v2.py:26–40` 对全部 cohort 生成并保存实际使用的 state、score 与 trace；`RewardScorer.score` 在第 34 行实际调用，`retrieval_metrics` 在第 54 行调用，`expected_hits` 在第 58 行用于同分平均。三种协议均写入 `retrieval_metrics.json` 和 `retrieval.pt`。

`analyze_comparison_v4.py:33–46` 汇总 greedy/sampled 与各数量子集，第 61–67 行计算所有成对区间，第 68 行执行全部采纳条件。全部 **72 个数量区间、18 个检索区间**由本次独立代码按显式图像 ID 重新配对计算，10000 次 bootstrap 和锁定种子均一致，最大差异 < 1e-12。三种检索协议的六项 Recall 和 mR 及每图聚类贡献也匹配。

当前最终执行链的统计函数均有输出证据。历史版本脚本保留为追溯材料，最终报告应以 v4 分析和 v3 完成记录对应的结果为准。

## E. 样本范围与统计解释：WARN

- 单数据集 **UCM**，一个 continuation seed；每臂 8 张 Agent 训练图，每图按固定次序重复 4 次，每组 8 候选，总计每臂 32 次尝试。
- 每策略 greedy 为 **273 图 = train 数量63图 + val210图**；sampled 为 heldout train55图 × 2 = 110 轨迹。全部数量标签为 train63 + val22 = 85 图。
- 锁定主指标为 B hard macro1to4，使用 heldout55 中真实数量为 1–4 的 **52 图**；分层数量为 **3、34、12、3**。真实数量为 7 的 1928.tif，以及数量为 6 的 2014.tif、2024.tif 保留在55图总体诊断，按锁定定义排除于主宏平均。
- 数量 hard 的候选集合来自同对象缓存模板。对象候选数量范围分别为飞机5、棒球场3、沙坑4、住宅3、储罐7、网球场6种数量。报告宜写“同对象全部缓存数量候选”。
- 数量 bootstrap 以图像为单位按真值分层，sampled 先在每图两轨迹内平均；检索 bootstrap 以验证图像及其5条描述为 cluster，固定图库。区间描述这一组策略及图库上的图像变异。
- 冻结读出器此前使用 UCM train 图文训练，并以 val210 duplicate-token-aware mR 从10个 epoch选取；`readout_reference_plan.json`、`inputs/parent_selection_lock.json` 与 `plan.json:161` 已披露。heldout55 的留出关系指 Agent 优化图像集合；val 检索属于既有选择集上的保护性诊断。
- 固定通用查询为 Describe the visible scene.，每图形成状态供后续数量模板/检索评分，见 `plan.json` 和 `prompt_design.py:6–21`。证据对应图像条件先验及冻结读出器表现。
- proxy_rank 实际31更新，pair/hard各32更新。可比较相同32次机会预算下的观察结果；严格等更新的三奖励归因条件当前为 false。

## F. 评估性质：PASS（分类明确）

| 项目 | 分类 | 解释 |
|---|---|---|
| 数量 A/B pair/hard | real_gt（人工标注子集） | 人工数量真值 + 固定文本模板 + 冻结读出器；指标衡量正确数量文本的相对排序 |
| val 图文检索 | real_gt | 原始数据集描述/图像归属；主协议额外采用重复 token 共享正例 |
| proxy 图像排名 reward | self_supervised_proxy | 以图像身份监督代理特征在训练图库中的排名 |
| 格式/工具消费率 | self_supervised_proxy | trace 的规则一致性检查 |
| 同图 reward–B margin 排序关联 | 代理机制诊断 | 人工真值约束下的冻结模型分数关系 |

数量评分适合描述冻结读出器对数量文本的辨别表现。本文范围中的“准确率”应始终带 A/B、pair/hard 及候选范围名称；Agent 生成回答的直接数数准确率需另行定义和人工核验。

## 关键数值与采纳条件

| 策略 | B hard greedy macro1to4 | 主协议 val mR | sampled 严格通过 | 实际更新 |
|---|---:|---:|---:|---:|
| baseline_continuation0 | 30.3922% | 76.9365% | 98/110（89.0909%） | 0 |
| proxy_rank | 29.0441% | 77.6667% | 110/110 | 31 |
| quantity_pair | 29.0441% | 77.6190% | 110/110 | 32 |
| quantity_hard | 26.9608% | 77.8095% | 110/110 | 32 |

主效应 **hard − proxy = −2.0833 个百分点，95% CI [−6.25, 0]**；hard − pair 相同。显式图像配对发现唯一正确性差异为 **2075.tif（真实数量3）**：proxy 的 B hard margin 为 0.0005775690，pair 为 0.0003533661，hard 为 −0.0005391240；由一个 12 图分层内错误贡献 −1/12/4。

主检索保护效应 **hard − baseline = +0.8730 个百分点，95% CI [−0.3651, +2.2381]**。其下限高于预设 −2 个百分点阈值。区间包含零，应以“通过预设检索保护条件”陈述。作为敏感性结果，strict_index 为 +0.5556 pp，CI [−0.9683, +2.1270]；strict_tie_averaged 为 +0.6559 pp，CI [−0.5158, +1.9828]。

| 预设条件 | 独立判断 | 依据 |
|---|---|---|
| 主数量效应95% CI下限 > 0 | FAIL | −0.0625 |
| 主检索效应95% CI下限 > −0.02 | PASS | −0.00365079365 |
| 全部四策略 sampled 严格率≥90%，自生成result≤2% | FAIL | baseline严格率89.0909%；四策略result均0 |
| 三训练臂同为32实际更新 | FAIL | 31 / 32 / 32 |
| 全部条件共同满足 | **false** | 与 analysis.json:10835–10865 一致 |

## 论文归因与处置

1. “完成三奖励继续短训和统一评估”有原始文件、更新和模型状态支持。
2. “数量困难负例奖励获得主数量优势”证据未达到预设要求；应如实报告负向点估计、包含零的区间及采纳失败。
3. “在已选冻结读出器上，通过预设val检索退化保护阈值”有本次数据支持，并保留验证集已参与读出器选择的说明。
4. “三个训练后策略在此次110条sampled轨迹中全部满足严格协议”有记录支持；总体采纳规则仍包含baseline。
5. 多数据集、跨seed稳健性、全流程未见图像泛化、严格等更新的奖励因果归因及原始RFPO实现历史复现等主张，留待对应实验单独验证。

按预设方案保留检查点并报告完成的负向/不确定比较。此次复核未启动追加训练、修改原始结果或对外同步。

## 追溯说明

首次本机断言额外要求历史人工台账中的句子与统一A模板逐字相同，因模板改写而触发；检查后确认85个数量与原对象均一致，脚本改为核对标签与 `original_object`，正确模板列的数量随后逐条重新验证。原始失败日志保存在 `fresh_reviewer_checks_initial_template_assertion.log`。这是独立核查脚本条件调整，所有实验材料保持原状。

技能包所引用的共享 reviewer-independence / experiment-integrity / review-tracing 文件在本机标准技能路径中未找到；本报告、原始任务信息、脚本、日志和输入哈希作为可追溯记录保留。该限制与同族审查状态一并披露。

