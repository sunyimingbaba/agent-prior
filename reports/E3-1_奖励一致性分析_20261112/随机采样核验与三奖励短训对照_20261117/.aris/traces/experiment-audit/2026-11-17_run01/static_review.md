# UCM 随机采样与三奖励继续训练静态审计

2026-11-17。同模型家族，暂定审阅（same-family/provisional）。总体 **WARN**。本次审查基于本地封存输入、代码与既有来源归档。本轮新运行的数值结论留待实际产物核验。

## 证据核查

### A. 人工标签与来源 · PASS

85/85图像、原对象名及真数量与人工工作清单逐条一致。170组A/B正负列均同模板、同对象，正列是真数量，负列是该图记载的不同数量。独立读取以往2100缓存原始归档，273/273个对应轨迹只有一个 model_requested_actual_tool 事件，反馈文本、置信度、接受标志及轨迹哈希与本轮cohort逐条相等。

证据见 inputs/quantity_cases.json:3；inputs/quantity_gallery.json:60；../图像依赖性与提示对照诊断_20261116/inputs/label_provenance/sources.json:3；../图像依赖性与提示对照诊断_20261116/code/prepare_inputs.py:33；code/reward_model.py:44。

### B. 分数及归一化 · PASS

代理奖励依严格超过目标的图像数取倒数排名；数量奖励是已单位化图文向量的正负余弦差。hard项在封存的同对象其他数词中取最大错误分数；组内优势标准化仅用于优化目标。试运行代码保留所有候选的原始分数和特征。

证据见 source_snapshot/vilt/modules/vilt_module.py:558；source_snapshot/vilt/modules/vilt_module.py:604；source_snapshot/agent/grpo/scorer.py:117；code/reward_model.py:40；code/run_pilot.py:22；plan.json:126。

### C. 本地材料及主张 · PASS

本地重算pilot清单31/31个条目，源代码快照77/77个条目。核对既有读出选择记录和适配器引用。本轮材料陈述计划，尚无训练或评估数值主张。

证据见 pilot_code_hashes.json:2；source_snapshot/inventory_before.json:2；inputs/parent_selection_lock.json:18；inputs/balanced_checkpoint_reference.json:3；README.md:3。

### D. 实际调用 · WARN

入口连接准备、真实采样和冻结读出。先收集实际请求后回放分类器原结果，将每个解码步末层末位置状态存档，末轮末步的同一h同时用于保存和打分。定义了独立教师重放函数；截至本静态快照，后续训练、概率误差见证、回滚和最终评估尚缺本轮产物。

证据见 code/run_initial_stage.py:10；code/run_pilot.py:21；code/stochastic_rollout.py:20；code/stochastic_rollout.py:31；code/stochastic_rollout.py:43；code/policy_logprob.py:9。

### E. 范围与归因 · WARN

UCM，8图，单继续训练种子，共同balanced32起点。主52图的真数量1、2、3、4的分母为3、34、12、3，另3图为6或7。主52图为本轮8张策略训练图之外的UCM训练图；读出曾在训练分割优化并按验证集选轮次。既有阶段已观察train63/B。数量A与B共享人工标签、冻结读出与既有候选集。实际更新数不同时报告固定机会预算的臂间结果。

证据见 plan.json:15；plan.json:147；plan.json:161；plan.json:167；readout_reference_plan.json:29；source_executed/sft/plan.json:80；source_executed/rl/plan.json:92。

### F. 评价类别 · PASS

计数为 human_eval 来源的人工已确认标签，B指标是冻结读出以真实数量为正文本的自动排序正确率。代理图库以数据图像行ID为目标，目标评分属于 self_supervised_proxy；协议合规则为输出格式诊断，分类器回放属于模型侧反馈。规划完整验证检索为 real_gt（数据集提供图文对应）。

证据见 code/reward_model.py:41；code/stochastic_rollout.py:31；code/matched_common_v3.py:104；code/matched_common_v3.py:124；plan.json:148。

## 实际产物核验事项

### 校准偏移身份

实际校准offset来自外部path/tensors.pt的闭包，模型state_dict哈希不覆盖该张量。两图精确旧状态复打可以验证读出点值。逐臂核实际注册offset与封存inputs/frozen_path_assets.pt内offset的数值哈希及读出状态哈希。

证据见 code/matched_common_v3.py:65；code/matched_common_v3.py:77；code/reward_model.py:17；code/reward_model.py:51；pilot_code_hashes.json:20。

### 训练代理图库全量对齐

图像、文件名均按train Arrow同一行读取，对被评分行核文件名。固定取前1680行，当前入口记录Arrow哈希和特征数。核实际图库Arrow是否等于下列权威哈希，以及1680个唯一有序文件名、特征有限且单位化。

证据见 code/reward_model.py:15；code/reward_model.py:23；code/reward_model.py:28；code/reward_model.py:43；../图像依赖性与提示对照诊断_20261116/inputs/dataset_sources.json:2。

### 基座与贪心基线复用

外部Qwen基座及处理器可与既有分片、配置及分词器哈希逐项核对。现有两条warm witness仅对旧h再评分；若复用旧贪心基线应另在当前生成器与提示下重跑锚点，核各轮生成token、末h、严格质量和读出特征；也可直接新生成。

证据见 code/milestone_common.py:68；code/prepare_stage.py:11；code/prepare_stage.py:22；plan.json:153；../纯推理先验完整生成/输入与模型清单.json:12。

### 采样分布、概率重放与真实步数

T=1真实generate，无top-k或top-p截断，每步比较处理后logits与仅首位屏蔽EOS的原logits。教师重放用实际assistant输出和该轮真实prompt。运行期核采样token重放的最大与平均绝对误差、恢复临时更新、全体新优化器及共同LoRA hash。每臂逐组核32次×8候选、保留失败和零方差组、跳步时参数和优化器状态一致、实际更新数及最终检查点；报告重放覆盖范围及误差。

证据见 code/stochastic_rollout.py:20；code/stochastic_rollout.py:24；code/stochastic_rollout.py:42；code/policy_logprob.py:9；plan.json:127；plan.json:131；plan.json:167。

### 分类器回放、同对象负例与评估解释

收到完整scene_classify请求后，真实历史分类结果进入独立用户回合；consumed的含义是结果进入随后生成的上下文。A/B均计算但仅A取入数量奖励，后续优化核读取所分配的奖励标量。困难负例为固定缓存中同对象其余数词，数词范围随对象而变。B检验措辞转移，主指标衡量读出排序；报告各真数量分母。

证据见 code/stochastic_rollout.py:31；code/stochastic_rollout.py:42；code/prompt_design.py:11；code/reward_model.py:44；plan.json:72；plan.json:147。

### 历史计划可读性

cache_plan沿用了父阶段2100缓存、test及四读出训练的历史字段。报告以当前plan与实际入口、运行日志计数，并在交付时明确cache_plan的父阶段引用用途。

证据见 cache_plan.json:39；cache_plan.json:55；plan.json:147；plan.json:171；code/run_initial_stage.py:10。

原始训练 Arrow 的 SHA-256 为 `4dd3c7eed39d309ffba1e77066a7fffae7912768c0a01f18fb578f3e6e5f967f`。原2100缓存归档中，1680个训练文件名按 row 升序、以紧凑 UTF-8 JSON 数组编码的 SHA-256 为 `4157799a7d5bb0d42462127267ecabfe0ca3882e175c91a49756617b1edc3d11`。
