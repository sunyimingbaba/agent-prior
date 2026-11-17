# 三奖励继续训练 V2 静态复核

2026-11-17。**总体 WARN**，same-family/provisional。核验范围为代码、历史输入与对当前阶段的预先承诺，未使用本轮新的训练或评价数值。

## A–F 复核

### A. 数据与冻结评分身份 · PASS

V2按权威train Arrow哈希、1680行、完整文件名顺序、所有特征有限且单位化验证代理图库。从已注册闭包抽取1024维offset与封存offset逐元素比对。准备阶段按15个权威基座/处理器文件校验，并核85个人工标注图像的像素身份。

证据见 code/reward_model_v2.py:32；code/reward_model_v2.py:40；code/reward_model_v2.py:45；code/prepare_stage_v2.py:15；inputs/dataset_sources_reference.json:2。

### B. 奖励与检索公式 · PASS

每次生成给三个奖励使用相同的h和读出；数量分数按56条固定文本的余弦算出，只有A写入训练奖励，B保留给评价。代理奖励和数量成对、最难负例奖励可从逐条原始分数重算。三种mR均对六个方向/截断值等权汇总，精确并列下均匀抽取的命中概率通过独立组合数例证（75个值，最大差0）。

证据见 code/reward_model_v2.py:57；code/reward_model_v2.py:64；code/train_reward_arm.py:36；code/evaluate_policy.py:53；code/tie_metrics.py:4。

### C. 源码清单与历史证据 · PASS

V2试运行清单38/38项、继续阶段清单6/6项通过本地字节哈希核验。15项基座引用与既有完整清单相同。上轮静态审计中的既有人工标签和工具反馈证据沿用。V1中两个算术路径的分数存在微量舍入差；V2要求图像特征逐位相同，并以1e-6为数量分数绝对误差阈值。

证据见 pilot_code_hashes_v2.json:2；continuation_code_hashes.json:2；execution_addendum_v2.json:3；code/prepare_stage_v2.py:26。

### D. 调用和更新证明 · WARN

已接入概率门、贪心锚点、三臂训练和四策略评估。只在提示选择成功且概率门通过后进入训练，各训练臂在独立进程内重新载入balanced32并新建AdamW。实际通过次数、有效更新及各评估结果须以运行产物核实。队列在评估完成后标记分析待做。已另封存分析脚本，其中实现了主终点、区间与采纳判定逻辑，实际调用和产物待核。

证据见 code/run_continuation.py:14；code/run_continuation.py:21；code/run_continuation.py:24；code/train_reward_arm.py:18；plan.json:156。

### E. 主指标选择与覆盖范围 · WARN

一个UCM继续训练种子、8张策略训练图，主要数量诊断为策略训练图外的52张train图，四个数量层分母3/34/12/3。每策略安排273条贪心和55图×2条随机轨迹。analysis_protocol_lock.json已指定主B数量指标用greedy，主检索警戒用repository_duplicate_token_positive，敏感性统计另列。封存时点待与实际运行时间线核对。训练预算相同而真实更新数不同时，归因限定在同等采样机会之下。

证据见 plan.json:147；plan.json:161；plan.json:164；code/evaluate_policy.py:25；code/evaluate_policy.py:55。

### F. 评价类型 · PASS

人工确认数量作为标签，对应冻结读出模板排序的诊断。代理奖励的图像行ID真实，排序目标为self_supervised_proxy。检索使用数据集图文配对（real_gt），在验证域评估。分类器为真实历史反馈，按本次模型实际请求回放；当前协议指标衡量格式与上下文暴露。

证据见 code/stochastic_rollout.py:31；code/stochastic_rollout.py:42；code/reward_model_v2.py:63；code/evaluate_policy.py:41。

## 运行产物验收要点

### 概率重放与覆盖

采样保存实际生成token的处理后分数并核实仅首位置禁EOS；教师前向把实际生成序列去掉最后一个输入token，在prompt末位开始逐token对齐（含终止token），同样屏蔽首位EOS。预检只抽固定排序前2条，来自同一训练图；后续每个有更新的组对全部候选的全部回合核对逐token误差（最大0.25 nat、每轮平均绝对0.05 nat）。仅预检另外检查末隐藏态余弦≥0.999。零方差组跳过教师重放；报告时给各阶段实际样本与token分母。

证据见 code/stochastic_rollout.py:20；code/policy_logprob.py:9；code/probability_gate.py:11；code/probability_gate.py:19；code/train_reward_arm.py:42。

### 临时更新恢复与失败记账

正常路径要求临时更新改变适配器哈希，再在finally按原适配器逐值还原，清梯度并核原哈希。临时进程的AdamW状态随后丢弃，三臂从新优化器开始。如果opt.step之后的断言或冻结检查失败，结果temporary_updates=int(ok)可能报0，实际应独立记step是否执行。发生该分支时以恢复与故障证据逐项核账。

证据见 code/probability_gate.py:27；code/probability_gate.py:32；code/run_continuation.py:21；code/train_reward_arm.py:21。

### 固定机会数和真实更新

各臂相同图像顺序和种子公式。每组8条完整采样含失败轨迹；仅本臂r_字段进入目标，组内使用FP32样本标准差，std≤1e-8时整个optimizer.step跳过，并检验参数哈希不动。每次真实更新后要求哈希变且梯度有限非零。验收时用逐条原始奖励和相同FP32规则重算，核32行组记账、256个记录哈希、Adam状态步数和最终权重。若计算某组中途报错，failure.json目前未写入该组已经算过的概率偏差列表，应保留组、样本/回合及误差以便复核。

证据见 code/train_reward_arm.py:32；code/train_reward_arm.py:38；code/train_reward_arm.py:45；code/train_reward_arm.py:48；code/train_reward_arm.py:51。

### 基线锚点、读出、验证图库

若选择contract_only，基线用一张train和一张val实际重新生成，以各轮prompt和输出ID、全部步状态、质量、图像特征和数量分数验证复用条件；失败则新生成。基线复用的273条仍按本轮读出重新打分。评估聚类贡献直接reshape(210,5,3)；新增code/verify_val_protocol.py:5以权威哈希及每图5句保护这一假设。验收核实际validation_identity_guard、封存时点及owner顺序。

证据见 code/baseline_replay.py:10；code/baseline_replay.py:16；code/baseline_replay.py:20；code/evaluate_policy.py:28；code/evaluate_policy.py:44；code/evaluate_policy.py:50；code/evaluate_policy.py:60。

### 主要数量终点与采纳口径

新analysis_protocol_lock.json:3–9已指定唯一口径。code/analyze_comparison.py:43–57按图先汇总2条随机轨迹，按真数量1–4四层对图像配对bootstrap，再以六个检索召回值汇总固定图库的逐图贡献。采纳要求主B hard差CI下界>0、验证mR差下界>−0.02、全部策略随机协议过门且三臂均32次实际更新。验收须重算各层分母、数量及检索CI，并核同步的封存时点。

证据见 plan.json:147；plan.json:164；code/evaluate_policy.py:25；code/evaluate_policy.py:55；code/run_continuation.py:24。

### 训练候选相关性的重采样单位

训练相关性在同一attempt和图像内计算正确。区间随后按有可定义相关性的attempt组独立重抽，字段名为spearman_image_bootstrap95；同图最多4次，实际独立图像仅8张。改作按唯一图像整簇重抽、携带其全部尝试，或仅报告描述统计；即便按图修正，范围仍为单次策略训练轨迹。

证据见 code/analyze_comparison.py:16；code/analyze_comparison.py:22；code/analyze_comparison.py:32。

### 随机评估的实际分母及排序

sampled模式只含55图，但会写出all85字段；内部images记录实际55。需为此模式标明实际覆盖。评价端目前仅汇总数量得分和协议，候选排序相关性仅针对pilot或训练；如要报告持出55图的随机排序，按每图两条保存记录统计奖励与B效用同向、反向、并列及相应分母。

证据见 plan.json:151；code/analyze_comparison.py:25；code/analyze_comparison.py:32；code/analyze_comparison.py:38。

补充核验已验证分析支持清单3/3个哈希，并复核了分析脚本。锁定时点与校验运行证据待同步。
