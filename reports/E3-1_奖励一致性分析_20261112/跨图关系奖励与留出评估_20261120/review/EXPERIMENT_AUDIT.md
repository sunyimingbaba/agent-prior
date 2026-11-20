# 实验完整性审计

日期：2026-11-20。审计任务：`/root/relation_transfer_integrity_review`。数值复核协作者：`/root/relation_reward_results_audit`。

**总体结论：WARN。确定性数值核验：PASS（以最终 stable 整库产物为准）。** 本审计由 fresh Codex reviewer 直接阅读代码和原始产物完成，`review_independence: same-family`，`acceptance_status: provisional`。当前审计只新建本目录的两个审计文件。标签有效性与整体科学结论保持 provisional。

最终交付报告的 SHA-256 为 `d5501d618d309e79196054ca5d7adc57fa0d7af4b934b883b5d6bc730f3458ce`。原始首版整库结果的数值复算差异已通过新建稳定结果解决；主实验、训练决策及补充关系实验结果保持一致。

## A. Ground truth 来源与划分独立性：WARN

- 原生检索正例取自 UCM Arrow 的图像归属与 caption 列；代码保存原始查询、行号、RGB 哈希、Arrow 哈希及目标 token 索引。证据：`code/shared.py:20–27`，`code/score_split.py:16–24,32–40,77–87`，`train/binding.json:4–18`，`val/binding.json:4–18`。
- 主组方向题与双物体题由助手依据真实像素构造，再由同家族助手进行评分前视觉复核，属于 **synthetic_proxy（自定义助手像素标签）**。原生 caption 检索属于 **real_gt**。报告已交代标签来源及人工签核情况。证据：`实验交付.md:9`，`review/像素复核.md:3–7,74–80`，`object_inputs/pixel_review.json:1–62`，`code/design.py:25–54`，`code/object_cases.py:7–41`。未来做正式性能主张时应补独立人工关系标注。
- 已独立检查 60 幅主图（train 44、val 16）和 12 幅补充候选的图号、RGB 哈希唯一性，两组交集为 0，与历史 excluded 清单交集为 0。主组 30 对各含两图，同场景、同 split，四套正反句逐字互换；补充组按评分前裁决保留 11 图。证据：`inputs/retained.json`，`inputs/excluded.json`，`object_inputs/cases.json`，`code/seal.py:19–25`，`code/object_cases.py:31–40`。
- 主组和补充组同场景配准筛查分别 5,395、1,065 次，记录检出数均为 0；筛查覆盖本项目 UCM train/val。地理位置独立性仍需空间元数据或来源地块核验。证据：`画面重叠筛查.json:1`，`object_inputs/画面重叠筛查.json:1`，`code/overlap.py:21–34,43–67`。
- 本阶段留出指新奖励选择和训练决策层面的留出。历史 LoRA、检索器、代理评分器各自已有训练/验证历史，报告末段已交代。验证评分入口检查训练决策文件，主方案和补充方案均有内容绑定。哈希确认冻结内容一致；时间先后依据保存的脚本顺序与执行记录。证据：`inputs/protocol.json:4–9`，`inputs/native_reward_comparison.json:2–7`，`object_inputs/plan.json:1`，`code/run_holdout.sh:6–16`。
- 原生描述补充奖励使用首条正例及偶数哈希负池，评价使用其余正例与奇数负池；相同 CLIP token 合并、首句所有 token 等价形式被剔除。训练/验证有效图数分别为 44/16，负池 token 计数为 798/837 和 201/190。已逐候选核对。证据：`code/native_reward.py:12–39`，`train/disjoint.json`，`val/disjoint.json`。

## B. 分数归一化、并列与统计分母：PASS

- A+B 使用两个组内 midrank 的等权平均作为选择规则；主准确率的分母为候选数、图数和预设图对数，奖励/准确率没有除以预测最大值来放大结果。余弦向量单位化属于相似度定义。原 RFPO 奖励为格式分加 `1/(1+rank0)`，原始 margin、rank0、格式分与总分均保留。证据：`code/analyze_split.py:29–48,55–64`，`code/score_split.py:69–74`，`CUP/agent/grpo/reward.py:22–44`。
- 同图随机候选按八条轨迹均匀取期望；最高分并列均匀取期望。主组按自然图对进行 20,000 次 percentile bootstrap，双视图对象组按来源图进行 bootstrap；独立复算均一致。证据：`code/analyze_split.py:12–22,55–86`，`code/object_readout.py:48–67`。
- 原生整库 RR、去首句 RR 与分离负池 RR 的所有相关候选中，目标与负例的精确同分计数均为 0，严格大于号在本批数据上未造成并列偏高。证据：`code/analyze_split.py:45–48`，`code/native_reward.py:19–22`。
- 双视图随机方式已显式区分。`object/summary.json:47–64` 与 `val/rotation.json` 的旧 `both_random` 使用同 k，AB 使用两侧独立并列；统一比较应取 `object/paired_choice.json` 或 `val/paired_choice.json` 的 `independent`/`same_draw` 对应字段。交付表使用独立选择 11.36% 对 22.73%，该口径及区间已复算通过。证据：`code/object_readout.py:53–59`，`code/analyze_split.py:101–107`，`code/paired_choices.py:7–11,19–40`，`object/paired_choice.json:20–37`。
- 完整图库采用期望边界并列；random、old、AB 各 64 次共同随机抽样，greedy 为单次固定读出。该整库均值是 Monte Carlo 估计，抽样表共 193 行；报告已说明 64 次抽样和常规 top-k 计分的区别。证据：`code/full_val_stable.py:8–19,42–58`，`val/full_caption_stable_draws.csv`。

## C. 结果存在性、原始证据与数字匹配：WARN（首版整库差异已修正）

全部指定输入、执行代码和结果文件存在，最终报告链接可解析。已核对 1,458 条实际生成轨迹的序列化文件 SHA、向量字节 SHA、bf16/3584 维、有限性、种子公式及工具事件计数；主分析纳入 1,188 条，补充纳入 198 条，其余 72 条属于预筛剔除图。60 条原始首句贪心记录报告归档逐位相同；16 条验证原图读出与保存的整库归档特征也一致。主组和补充输入锁、绑定文件、训练决策引用的 summary/protocol、原生奖励补充方案绑定均通过。服务器 Arrow、模型文件和历史归档的实物核验由原执行日志及绑定提供；本次独立计算覆盖交付向量、原始生成记录与统计产物。

| 独立核验项目 | 范围与结果 |
|---|---|
| 图文 margin 与原奖励 | train 792 行、val 396 行、object 198 行；主组 A–D 最大误差 8.327×10⁻¹⁶，补充组 6.661×10⁻¹⁶；主组 proxy rank0 全部精确吻合 |
| 主表与严格子集 | images.csv、summary.json、strict.json 的均值及 20,000 次聚类区间全部一致 |
| 奖励相关性 | train/val 的逐图 Spearman、有效分母及图对 bootstrap 区间全部一致 |
| 原生首句奖励补充 | reward_margin、eval_margin、eval_RR、relation_rank、joint_rank 与全部汇总区间一致 |
| 双视图选择 | independent/common-quantile coupling 的逐图值、分组均值及区间一致 |
| 像素/先验交叉替换 | 216 行四格分差及全部聚类区间一致；最大偏差 1.735×10⁻¹⁷ |
| 稳定整库结果 | baseline 6 指标及 mR、193 行×7 指标、4 个 selector 汇总均一致，最大误差 0.0 |

**本次发现并解决的数值事项：** 首版 `val/full_caption.json` 的 baseline mR 为 0.6002964561893134；从交付特征在独立环境复算得到 0.6005540777564587。差异集中在 I2T 三项，T2I 三项一致。重复描述展开后的精确同分对浮点计算路径敏感；首版实际 score 矩阵未保存，其平台差异的细粒度原因保留为推断。新建 `code/full_val_stable.py` 先对 391 个唯一 token 组按固定维度顺序累加点积，再展开为 1,050 条带归属描述；保留首版文件，新结果逐项独立复算通过。交付文字现采用 stable 值：archive 60.06%、random 58.62%、old 59.03%、AB 59.05%、greedy 59.48%。其中 AB−old 为约 0.0240 个百分点。

`val/stable_cross_device.json` 记录执行者提供的服务器/本机结果哈希；本审计独立确认本机文件匹配：stable JSON 为 `d73f20caec51bcd537a9b08d346addf1a922e8e2697faecc2161f771e33dbb3d`，逐次 CSV 为 `502b27573376114404e4fbacfbfa042dc414e00eb4c7be1490ac9c75b70f1f78`。证据：`code/full_val.py:35–50`，`code/full_val_stable.py:25–38`，`val/full_caption_stable.json`，`实验交付.md:61`。

## D. 实际执行、死代码与工具协议：PASS

- 生成、读出、分析及补充指标的函数均有入口调用，原始产物和日志支持执行。train.log:61 记录 792 行/44 图；val.log:33 记录 396 行/16 图，val.log:50 为 VERIFIED；generate.log:78 与 objectgen.log:28 分别记录 62 与 12 个源图生成结束。证据：`code/run_stage.sh:7–9`，`code/run_holdout.sh:9–16`，各脚本的 main 入口。
- 已部署 LoRA、冻结下游检索器和纯推理重建代理均有哈希；`CUP/agent/rollout.py`、`grpo/reward.py`、`grpo/scorer.py` 当前文件 SHA 与 train/val binding 完全一致。评分使用冻结模型，训练门槛为五项中三项通过，`pass_gate=false`。交付只陈述冻结候选选择，未报告策略短训提升。证据：`code/readouts.py:12–21`，`code/analyze_split.py:88–94`，`train/decision.json:2–11`。
- 审计检查了工具阈值差异：本轮直接回填 top-3，与历史 pipeline 的 `--tool_conf_threshold 0` 一致。已评分轨迹共 690 次工具事件，其中 170 次置信度小于 0.29。CLI 默认阈值为 0.29，交付已明确历史实跑阈值 0。证据：`code/generate.py:37–44,65–78`，`CUP/agent/rollout.py:231–232,273–277`，`../EOS训练验证分歧诊断_20261114/inputs/historical_pipeline_domain_tool.sh:29–33`，`实验交付.md:11`。
- `readouts.scorer_for` 与 `shared.locked` 为未被当前入口调用的辅助函数；实际冻结检查和 scorer 建立在各评分入口显式完成。未发现已宣称指标只定义而未生成相应结果的情形。

## E. 规模、范围与结论：WARN

实际范围为单一 UCM 部署，44 幅训练图/22 对、16 幅本阶段留出图/8 对，预先固定的严格子集 12 图/6 对；补充双物体组为 11 个来源图，含 9 个实体关系与 2 个对象—阴影关系。每图每条件八次候选随机生成加一条 greedy，属于候选采样重复。训练决策未触发新策略更新。完整图库结果只替换 210 图中的 16 图，同时改变了相应查询输入。

当前报告保留了宽区间、措辞敏感性、高置信子集净差为零、原生检索保护条件未通过以及补充实体/阴影数量小的说明，结论与现有证据相称。后续选择奖励的建议属于候选方案，新增训练效果和跨数据集表现仍待实验。证据：`inputs/protocol.json:4–9`，`train/decision.json`，`val/strict.json`，`object/paired_choice.json:56–109`，`实验交付.md:17–25,31–43,57–63`。

## F. 评估类型：WARN（混合来源已披露）

| 评估 | 分类 | 来源 |
|---|---|---|
| 原生 caption RR、分离正负描述评价、完整图库召回 | real_gt | UCM Arrow 的真实图像归属和原描述；完整图库采用自定义期望并列规则 |
| A–D 方向题、双物体题、旋转题 | synthetic_proxy | 助手根据真实像素构造并经同家族评分前视觉复核；旋转沿用几何标签交换 |
| 原 RFPO 候选排序 | real_gt target identity + model-based proxy reward | 真实目标图行号，冻结重建评分器代理余弦排名和格式项 |
| 像素/先验交换分解 | synthetic_proxy diagnostic | 真实原图及旋转图、冻结隐藏向量、自定义方向描述 |

本批方向标签的签核主体为模型助手，人工评价状态为未新增；图像来自真实 UCM 照片及确定性旋转。原生数据集标签与自定义辅助标签在当前交付中有来源说明。

## 后续读取与主张影响

1. 完整图库引用 `val/full_caption_stable.json` 和 `val/full_caption_stable_draws.csv`；`val/full_caption.json` 保留为首版历史记录。
2. 双视图联合正确率显式注明 independent 或 same_draw，统一从 `paired_choice.json` 读取。
3. 正式关系性能主张继续保留自定义助手标签、评分前同家族复核、本阶段留出、单 UCM 冻结候选和 provisional 信息；新增人工标签与新数据集能提高证据独立性。
4. “本轮保留原策略权重”得到五项预设门槛和已执行产物支持。“方向奖励普遍稳定提升检索”尚待进一步实验；当前报告已采用局部有效、变化敏感的描述。

完整输入哈希清单、确定性核验摘要及修正记录见同目录 `EXPERIMENT_AUDIT.json`。
