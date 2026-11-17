# 代理排名奖励臂及跨设备标准化复核

2026-11-17 · fresh same-family / provisional。单臂训练记录与跨设备验证修订 PASS；全实验范围 WARN。代理臂第 12 次跳步，32 次机会、31 次真实更新；预注册的每臂 32 次真实更新条件已失效。

按本地实际输入与两份检查点重算；附带 JSON 列出 859 项输入 SHA-256。

## A · 参照来源 · PASS

该臂的目标是实际训练图像在固定 1680 行图库中的行标，格式奖取自保存的协议轨迹。独立以每个候选的 proxy_feature 和整座冻结图库重算点积，CPU FP32 和 FP64 均对 256/256 条恢复相同的严格高分计数。数量 A/B 的正列和错误数词使用事先确认的人工数量；以封存的逐条特征及原始分数独立复核了各辅助奖励。

证据行号　`code/reward_model_v2.py:32`，`code/reward_model_v2.py:60`，`code/reward_model_v2.py:63`，`code/reward_model_v2.py:67`，`code/verify_raw_cpu_v2.py:26`，`code/verify_raw_cpu_v2.py:34`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/samples.jsonl:1`。

## B · 实际优势与高精度参照 · PASS

从原始样本奖励重构训练实际使用的每组 8 个 FP32 输入值；CUDA 见证在 A100、Torch 2.4.1+cu124，以独立脚本从 256 条奖励重算均值、无偏标准差、优势，逐组绑定两份原 JSONL 的 SHA-256，32/32 组的均值/标准差及 256/256 个优势逐值相同。本机无 GPU 复跑，审核已归档的 CUDA 见证，另外独立执行 CPU 检验。用 CUDA 统计量在本机逐元素计算实际 FP32 优势，与记录的最大差为 0。旧法从头做 CPU FP32 归约，在 23/32 组越过 1e-5 的优势差，最早第 3 组 0.0001885891，第 28 组达到 0.12509894。CPU FP32 与 CUDA 的均值、标准差最大差为 1.49011612e-8、7.45058060e-9。将相同的 FP32 输入提升到 FP64 再求数学均值/标准差，与 CUDA 统计量最大差为 1.30385160e-8、6.04742085e-9；两者按 1e-8 得到完全相同的 31 次更新、1 次跳步。FP64 数学优势和实际 CUDA FP32 优势的最大绝对差为 0.00888412（第 25 组）。第 26 组记录优势的组均值绝对值达 0.00509666，反映舍入放大。修订额外约束真实 CUDA 逐值一致、CPU FP64 的统计量误差、跳步判定及真实损失；8ε×maxabs 仅作为归约绝对误差容限，观察到的误差至多占该容限 6.82%。该臂的微小代理奖励差会放大标准化的舍入误差，固定预算结论对应封存的 CUDA FP32 训练实现。

证据行号　`code/train_reward_arm_v2.py:38`，`code/verify_training_cpu.py:20`，`code/verify_training_cpu.py:21`，`code/verify_normalization_cuda.py:9`，`code/verify_training_cpu_v2.py:20`，`code/verify_training_cpu_v2.py:21`，`normalization_verification_addendum.json:3`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/independent_cuda_normalization.json:5`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/independent_cuda_normalization.json:637`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/independent_cuda_normalization.json:715`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/updates.jsonl:3`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/updates.jsonl:25`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/updates.jsonl:26`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/updates.jsonl:28`。

## C · 完成臂、哈希链与检查点 · PASS

32 次固定采样机会，8 张训练图各用 4 轮，每次 8 条，共 256 个不同样本 ID、轨迹文件、张量文件。246 条严格完成，8 条格式失败、2 条混合/未知工具输出；自行生成结果标签 0 条。逐行核共同 balanced32 起点、封存图序、采样种子、先后状态哈希及真实更新计数。第 12 次（124.tif）8 项真实奖励和其 FP32 表示分别组内相等，排名均为 1675，严格完成均成立；FP32 奖励为 0.2005966604，标准差为 0，策略哈希原值延续。其余 31 次切换哈希；最小非零标准差 1.26440540e-7（第 20 组）。对两个 checkpoint 各 16 个完整文件重新验字节数和 SHA；从每份实际 adapter 重算 584 个 FP32 张量按名排列的状态哈希，匹配第 16/32 次机会后的日志。两个 optimizer.pt 各有 584 个不重复参数状态、95178752 个参数元素；所有 exp_avg、exp_avg_sq 有限，二阶矩非负，各状态 step 在两个检查点分别一致为 15、31。训练状态中的计划及提示 SHA 也和输入相同。两个实物之间的逐步哈希来自运行日志，实际权重在两个存档点按字节独立核验。

证据行号　`已完成代理组复核/原始运行记录/training_v2/proxy_rank/result.json:3`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/settings.json:68`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/samples.jsonl:89`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/samples.jsonl:96`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/samples.jsonl:256`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/updates.jsonl:12`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/updates.jsonl:16`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/updates.jsonl:32`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/intermediate_16_checkpoint_manifest.json:4`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/final_checkpoint_manifest.json:4`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/intermediate_16/checkpoint/training_state.json:3`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/final/checkpoint/training_state.json:3`，`code/verify_training_cpu_v2.py:33`，`code/verify_training_cpu_v2.py:39`，`已完成代理组复核/原始运行记录/independent_completed_arm_verification_v2.json:8`。

## D · 回算、损失与修订范围 · PASS

核原始 392 条记录、778 轮，分别为本臂 256、预试验 128、旧失败组 8 条；本臂新生成 510 轮，其中有更新的 494 轮、16971 个 token 与记录的生成 token 概率逐值一致，零方差组另有 16 轮。对每次有效更新按“负优势 × 各轮有效 token 的对数概率和 / 8”累计损失；独立 FP64 累计与训练原值最大差 4.795e-6（第 4 组）。复核原始 CPU 核验主体和针对代理臂的训练 CPU 核验主体，复算结果与已有两份验证 JSON 相同。修订前后的 diff 只改变归约验证与见证对应关系，原始训练源码哈希、初始 dtype、奖励公式、概率阈值、实际更新规则和预先采用门槛均与旧清单相符。旧 CPU 验证器也与原已封存哈希一致。新后处理入口连接独立 CUDA 归约重放及后续 CPU 核验。

证据行号　`code/train_reward_arm_v2.py:42`，`code/train_reward_arm_v2.py:43`，`code/verify_training_cpu_v2.py:24`，`code/verify_training_cpu_v2.py:28`，`code/verify_completed_arm_cpu_v2.py:14`，`code/run_post_queue_v2.py:14`，`cached_continuation_hashes.json:4`，`post_code_hashes.json:3`，`post_code_hashes_v2.json:2`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/updates.jsonl:4`，`已完成代理组复核/原始运行记录/independent_raw_verification.json:3`，`已完成代理组复核/原始运行记录/independent_completed_arm_verification_v2.json:25`。

## E · 预注册门槛与归因 · WARN

代理臂实际更新 31 次，已确定“每臂均达 32 次实际更新”这一预注册采用条件为 false。另两臂实际更新数、完整四策略评估、数量主比较与检索区间均待同步。若其他臂更新数不同，应在固定机会预算下比较各自最后检查点，并报告跳步机制和差异。训练运行的 CUDA FP32 标准化已由同设备原始记录佐证，跨归约实现标准化存在数值敏感性；报告使用真实日志优势及每臂实际更新数。UCM 八图、一个继续训练种子限制跨设置推广。

证据行号　`plan.json:125`，`plan.json:157`，`plan.json:167`，`analysis_protocol_lock.json:3`，`code/analyze_comparison_v3.py:68`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/result.json:4`，`已完成代理组复核/原始运行记录/independent_completed_arm_verification_v2.json:25`。

## F · 评价类型 · PASS

代理排名属于 self_supervised_proxy，以真实图像身份确定目标、用模型特征的相对排名提供奖励。规则化工具完成率是自动协议指标，原始概率及标准化重放属于计算一致性检查。人工数量支撑的读出排序与验证图文对应检索是独立的下游评价，当前快照呈现训练完成情况。

证据行号　`code/reward_model_v2.py:63`，`code/verify_raw_cpu_v2.py:26`，`code/evaluate_policy_v2.py:42`，`code/evaluate_policy_v2.py:55`，`已完成代理组复核/原始运行记录/training_v2/proxy_rank/result.json:13`。
