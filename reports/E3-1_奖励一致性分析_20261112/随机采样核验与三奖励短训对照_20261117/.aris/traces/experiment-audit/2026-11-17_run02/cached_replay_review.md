# UCM 三奖励短训，缓存重放专项复核

2026-11-17。fresh same-family / provisional。总体 **WARN**。本次读取封存源码、服务器运行库副本和本地概率修复快照。修复门控有数值依据，三臂正式成绩与采用结论待归档。

涉及运行记录的路径均以实验目录为基准，SHA-256 逐文件清单见同目录 JSON（共 348 项）。

## A · 人工数量及对应关系 · PASS

将 85 幅图像的文件名、原对象名、确认数量与来源逐一连接至人工工作清单。85/85 一致。逐个验证 170 对 A/B 正负文本的列、同一对象模板和数词，正列等于真数量，负列为另一数量。63 幅来自训练集、22 幅来自验证集。验证检索以 Arrow 图像行和各自的五条描述为源，生成器读取通用提示、图像及请求后回放的分类反馈。B 正确率衡量冻结读出给真数量文本的排序。

证据行号　`../图像依赖性与提示对照诊断_20261116/inputs/label_provenance/sources.json:3`，`../图像依赖性与提示对照诊断_20261116/inputs/label_provenance/100组工作清单.csv:99`，`概率修复复核快照/原始运行记录/inputs/quantity_cases.json:8`，`概率修复复核快照/原始运行记录/inputs/quantity_gallery.json:60`，`code/prepare_stage_v2.py:22`，`code/evaluate_policy_v2.py:41`，`概率修复复核快照/原始运行记录/validation_identity_guard.json:5`。

## B · 原始分数与归一化 · PASS

代理分量按 1/(1+严格高于目标的训练图像数) 计算；1680 行名称顺序与封存哈希一致。数量分量按同对象真数词分数减指定负例或最大错误数词分数计算，格式奖励为 0 或 0.2。组内标准差在优势计算使用；准确率分母来自标注图数。以快照重新执行 CPU 核验主体（只将写出改为和既存 JSON 比对），136 条记录、268 轮均复现，数量/代理点积最大绝对误差分别为 5.96e-8、2.38e-7。1024 维 offset 的原始字节 SHA-256 与读出契约相同；代码同时核对注册闭包、独立只读参数和状态哈希。

证据行号　`code/reward_model_v2.py:32`，`code/reward_model_v2.py:45`，`code/reward_model_v2.py:60`，`code/reward_model_v2.py:63`，`code/reward_model_v2.py:67`，`code/reward_model_v2.py:71`，`code/train_reward_arm_v2.py:38`，`概率修复复核快照/原始运行记录/assets/scoring_contract_v2.json:2`，`概率修复复核快照/原始运行记录/independent_raw_verification.json:3`，`code/verify_raw_cpu_v2.py:32`。

## C · 当前快照对应的主张 · PASS

预试验 128 条，旧训练首次尝试另存 8 条、实际更新 0。复核旧组 16 轮中仅样本索引 3、轮索引 1 超阈值；该轮 87 个 token，整段最大差 0.3643019199、均值 0.0271054246；缓存诊断两者均为 0。同一原始轨迹、目标 token、重放 logits 和隐藏状态可接回新版门控。新版 10 条/20 轮，829 个 token 的已选 token 对数概率均逐值相同；CPU 还对 20 轮共 60 个代表位置从教师全词表 logits 重算，最大算术差 6.99e-7 以内。20/20 末隐藏张量与原生成逐元素相同。直接持有双方全词表 logits 的是 2 轮 6 个位置，逐元素相同。运行说明使用“87词”，原始分母为解码 token，应写“87个生成token”。旧 interrupted_0 只同步了清单，在这份快照无法重哈希旧检查点本体。新版正式三臂结果、四策略评价、最终 CI 未同步。

证据行号　`概率修复复核快照/原始运行记录/training/proxy_rank/failure.json:3`，`概率修复复核快照/原始运行记录/training/proxy_rank/interrupted_0_checkpoint_manifest.json:3`，`概率修复复核快照/原始运行记录/training/proxy_rank/samples.jsonl:4`，`概率修复复核快照/原始运行记录/probability_failure_diagnostic/full_replay_rounds.json:53`，`概率修复复核快照/原始运行记录/probability_failure_diagnostic/cached_replay.json:9`，`概率修复复核快照/原始运行记录/probability_gate_v2/result.json:6`，`概率修复复核快照/原始运行记录/probability_gate_v2/rounds.json:6`，`code/run_pilot_v2.py:21`，`code/probability_gate_v2.py:19`，`阶段运行说明_20261117.md:7`，`阶段运行说明_20261117.md:15`。

## D · 实际调用、梯度与恢复 · WARN

已通过的入口使用本轮原始 prompt 重新编码并断言 token ID 相同；prefill 位置预测首个生成 token，随后每次附加前一步目标、更新原生缓存和掩码。每轮仅首位屏蔽 EOS；终止 EOS 计入损失，工具停止由已生成串触发。梯度记录路径连续：Qwen 将 k_proj/v_proj 送入缓存，再以返回 K/V 计算注意力；DynamicLayer 通过 torch.cat 保存历史张量，eval 模式关闭训练专用 checkpoint 分支，LoRA 在普通模块前向参与。detach 发生在报告、隐藏状态副本或优势上，标量 loss 使用保留计算图的选定 token 概率和。独立 CPU 微测采用实际源码中的 DynamicLayer 类，对零基第 2 个前向步的注意力标量求导。前缀 K/V 与当步 K/V 梯度均非零，与显式拼接参考逐元素相同。实际模型一次累计反传梯度范数 30.93355751，临时 AdamW 更新改变 LoRA 哈希；从 checkpoint 恢复时逐张量比对 584 个适配器，再核原始哈希。临时优化器本身并未回卷，它随门控子进程结束；每臂在新进程中重新播种、载入同一初始适配器并创建空 AdamW。已核 8/8 修复清单、5/5 后处理清单与 5/5 库快照条目。四策略、三臂和完整事后 CPU 检查入口已连接，当前正式调用结果留待逐臂产物。门控中 v.grad is None 也能通过；全模型完整参数梯度的数值等价尚无逐参数参照。

证据行号　`code/stochastic_rollout.py:18`，`code/stochastic_rollout.py:24`，`code/policy_logprob.py:3`，`code/policy_logprob_cached.py:7`，`code/policy_logprob_cached.py:9`，`code/policy_logprob_cached.py:14`，`code/policy_logprob_cached.py:16`，`code/milestone_common.py:76`，`runtime_source_snapshot/transformers_cache_utils.py:119`，`runtime_source_snapshot/transformers_cache_utils.py:776`，`runtime_source_snapshot/transformers_generation_utils.py:974`，`runtime_source_snapshot/transformers_models_qwen2_5_vl_modeling_qwen2_5_vl.py:646`，`runtime_source_snapshot/transformers_models_qwen2_5_vl_modeling_qwen2_5_vl.py:660`，`runtime_source_snapshot/transformers_models_qwen2_5_vl_modeling_qwen2_5_vl.py:816`，`runtime_source_snapshot/transformers_models_qwen2_5_vl_modeling_qwen2_5_vl.py:1557`，`runtime_source_snapshot/peft_tuners_lora_layer.py:509`，`code/probability_gate_v2.py:25`，`code/probability_gate_v2.py:27`，`code/probability_gate_v2.py:32`，`code/checkpoint_io.py:11`，`code/run_continuation_v2.py:17`，`code/run_continuation_v2.py:21`，`code/train_reward_arm_v2.py:19`，`code/train_reward_arm_v2.py:21`，`code/run_post_queue.py:14`，`概率修复复核快照/原始运行记录/probability_gate_v2/result.json:9`。

## E · 分母、配对和归因 · WARN

扩大门控 10 条均属 102.tif，所测策略是共同起点；8 张训练图是正式继续训练设计。每个有信息量的组中，每个样本每轮再次检查最大/均值误差后再反传；零方差跳过 AdamW。三臂固定 32 次机会、每次 8 条，每隔 8 组重复同一图序，采样种子相同，定取最终 checkpoint。不同实际更新数只支持在匹配采样预算下比较。本轮训练之外的 55 幅由真数 1/2/3/4/6/7 的 3/34/12/3/2/1 幅构成；主要指标仅前四层共 52 幅，各层等权，1、4 层各 3 幅。计划先对同图同策略的两条随机轨迹取均值，再按固定图序逐图作差并在各真数层内重采样；主检验使用贪心。验证检索固定图库，210 个图像簇各带 5 条描述，10000 次百分位配对区间。代码按顺序配对，最终归档应直接验证各策略键及同图种子顺序。55 幅在既往 SFT/RL 评价中已使用；冻结读出按该验证分割选取。图像重采样区间只覆盖当前固定策略和图库，八图单种子的结果暂待最终记录。

证据行号　`概率修复复核快照/原始运行记录/probability_gate_v2/rounds.json:5`，`概率修复复核快照/原始运行记录/probability_gate_v2/rounds.json:1024`，`code/train_reward_arm_v2.py:32`，`code/train_reward_arm_v2.py:42`，`code/train_reward_arm_v2.py:46`，`code/train_reward_arm_v2.py:51`，`plan.json:113`，`plan.json:147`，`plan.json:157`，`analysis_protocol_lock.json:3`，`code/analyze_comparison_v3.py:43`，`code/analyze_comparison_v3.py:65`，`code/analyze_comparison_v3.py:67`，`code/analyze_comparison_v3.py:68`，`code/verify_metrics_cpu.py:38`，`code/verify_metrics_cpu.py:56`，`source_executed/sft/plan.json:240`，`source_executed/rl/plan.json:92`，`readout_reference_plan.json:29`。

## F · 评价类型 · PASS

数量标签是 human_eval 来源的人工已确认事实；B 阈值排序为 real_gt（以人工数量标注为参照的自动评价）。数据集提供验证描述与图像对应，检索为 real_gt，主口径把同 token 描述扩充为正例。训练代理排名以真实图像行作目标，投影特征和训练图库产生的排名为 self_supervised_proxy。严格协议完成率为规则检查，历史分类器输出属于模型工具反馈。采样与重放为数值一致性检验，候选 B margin 相关性为读出诊断。这几项分别记录。

证据行号　`code/reward_model_v2.py:63`，`code/reward_model_v2.py:67`，`code/stochastic_rollout.py:31`，`code/stochastic_rollout.py:42`，`code/evaluate_policy_v2.py:44`，`code/evaluate_policy_v2.py:57`，`code/analyze_comparison_v3.py:20`，`code/probability_gate_v2.py:19`。

## 主张影响

- supported　该图下旧法的越阈值失败与新版门控修复。
- supported　门控一次临时优化后恢复 LoRA，门控子进程与正式臂的 AdamW 隔离。
- needs_qualifier　跨步 KV 计算图有源码、实际反传及缓存微测支持；该次微测覆盖缓存梯度，全模型逐参数数值对照尚待证据。
- needs_qualifier　扩大概率门控覆盖 102.tif 的 10 条候选，当前证据涵盖 829 个所选 token。
- unsupported　三个奖励的最终优劣、达标与稳定性尚无本快照的正式比较结果。

## 最终同步后待核项目

- 按三臂各 32 次机会恢复有信息量组、实际步数、跳步、LoRA 哈希链、检查点和优化器状态。
- 四策略逐图核（图像、生成模式、样本索引、种子）对齐后，从原始检索/数量分数、标注和固定图库重建区间与全部门槛。
- 后续报告注明单图修复门控、新旧样本分账、主分层分母及历史读出/验证选择。
