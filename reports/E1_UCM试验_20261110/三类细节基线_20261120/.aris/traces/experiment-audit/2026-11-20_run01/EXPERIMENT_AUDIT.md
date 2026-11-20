# E1 UCM 三类细节基线独立审核

- 日期：2026-11-20
- 审核者：独立 Codex 子代理 `/root/ucm_e1_baseline_audit`，继承模型；本次接口未提供可核验的精确模型名与推理档位。
- review_independence: same-family
- acceptance_status: provisional
- Overall Verdict: **WARN**
- Integrity Status: **warn**
- 写入范围：仅本审核目录。未调用服务器、训练模型或修改实验原件。

本轮已独立重算 6 组保存特征对应的 1,452 项分数，并核对原有 5 组汇总的 60 行计数、正确率、均值和并列情况。所有数值核验通过；最大 FP64 差为 5.551115123125783e-16。主要问题集中在 AGSE 的实际生成行为、标注来源、样本独立性和解释范围。

## A. Ground Truth Provenance: WARN

数量部分是 100 张图、每图 A/B 两种句式，共 200 项。`code/prepare_images.py:14–19` 从既有人工确认清单复制正确/错误描述；独立逐项比对 `../第十步_统一句式与对照方案/100组两种句式_对照清单.csv` 全部吻合。其数量判据有项目用户确认记录，未使用本轮待评价检索器输出来生成标签。

属性/关系部分由 `inputs/paired_visual_labels.tsv:2–22` 定义 21 组图像反向配对。`code/lock_cases.py:10–17` 明确把来源写成 `assistant_pixel_review`。这是助手依据真实图像新增的诊断标注，尚无独立人工复核记录。报告中应持续保留该来源，使用“助手逐图审阅的诊断样本”描述。

审核者亲自查看 `review/pairs_01.jpg` 至 `pairs_06.jpg` 中全部 42 个带文字图块，未发现明确的正负颠倒：草地/土质内场、草地/铺装停车面、弯曲/直线跨线桥、网球场周边颜色、沙坑形状、水面颜色、屋顶明暗、岸线左右、内外场左右、道路上下跨越、沙坑/道路左右、罐体与球场停车场上下位置均有可辨认的图像依据。两处保留语义精度提醒：250 图的内场由草芯、投手丘和土环组成；1486 图的上层道路曲率较轻。这两项目前支持对应句子的相对对比，独立人工复核时宜优先确认。

## B. Score Normalization: PASS

`code/score.py:73–80` 用图文 L2 单位特征的内积得到余弦，保存正句分数、负句分数、两者差值与正确标记；判断依据为 `positive_cos > negative_cos`。`CUP/vilt/modules/vilt_module.py:558–560,604–605` 的向量归一化是余弦定义中的标准处理。计分路径未使用模型分数的最大值、均值或范围作为性能指标分母。

独立重算核对了所有 1,452 项的正/负分数、margin、correct、tie、特征有限性及范数；原 baseline 汇总 60 行全部一致。证据与可重跑脚本：`independent_evidence.json`、`independent_check.py`。

## C. Result File Existence and Attribution: PASS

`outputs/*/{settings.json,rows.json,features.pt}` 六组均存在并可读，模型记录中的 cases、scored_images、features 哈希均相符。原 `score_verification.json:2–13` 覆盖五组主/衔接结果、1,210 项；新增 `AGSE_archived` 242 项属于补充条件，本审核另行重算，保留两份校验的覆盖范围。

方法归属有证据支持：

| 行 | 实际设置及证据 |
|---|---|
| CLIP | 冻结 ViT-L/14，视觉提示长度 0、文本无提示；`code/score.py:36–40,45–51,68–70`。 |
| CUP_CCB | 归档 UCM 聚类先验权重，`agent_prior=false`；`code/score.py:12,27–33,66–67` 与 `outputs/CUP_CCB/settings.json:7–12`。 |
| ASPF | 11 月 13 日重建的纯推理读出器，Qwen 基座、PLAIN_PROMPT、无 LoRA/工具；`code/common.py:11`、`code/generate.py:20–24,46–48`。 |
| AGSE | 历史 UCM 主 LoRA、工具协议和 version_2 读出器，本轮使用通用查询；`code/generate.py:20–35,46–48`、`outputs/AGSE/settings.json:7–12`。 |
| CLIP_legacy | 原 ViT-L/14 加 16 个全零视觉 token；独立复核旧数量 200 项判断 200/200 一致，margin 最大差 3.0736931139196955e-7。 |
| AGSE_archived | 同一个 AGSE 读出器，输入历史首条描述缓存；`code/score_archived.py:58–63`、其 settings 的 `prior_source`。 |

严格加载已在本轮计分路径实现：`code/score.py:29–32` 先清空构造器隐式加载路径，再执行 `load_state_dict(..., strict=True)`；`scoring.log:17,26,35` 均有成功完成记录。归档 README 第 54 行给出的 strict=False 是历史加载示例，本轮采用严格加载。`asset_bindings.json` 中 rollout、tools、prompt_clip/model.py、vilt_module.py 的哈希与本地四个源文件完全一致；归档权重哈希另见 `archived_checkpoint_hashes.json`。本审核执行了静态加载路径核查及保存特征复算，模型加载本身采用已保存日志作为运行证据。

CUP 原文 PDF 第 4 页图 2/III.A–B、第 6 页 III.D 给出的 PGM、RSP-CLIP、UEM 结构与当前 `基线来源.md:3–16` 基本对应。CCB 属于 CUP 的提示生成模块；当前命名“CUP（CCB），项目复现聚类先验版”准确。Agent 分支通过 `vilt_module.py:584–593` 替换视觉先验来源并继续提示融合；训练与生成路径均有差异，方法行比较的解释应涵盖这些差异。

## D. Dead Code / Execution: PASS

计分、特征导出及汇总都能与保存结果逐项对应；`code/summarize.py:35–51` 的各项指标在 CSV 中均有输出，`scoring.log:4,8,17,26,35` 记录对应运行完成。生成脚本的工具分支确实可被注册和调用；本轮通用查询的实际调用数量为零，应报告此观察。

`native_token_control.json` 现提供补充运行证据：133/133 通用查询首轮仅输出 EOS token 151645，且全部隐藏态与本轮保存先验逐元素一致。6 个原描述控制首轮均有其他 token，隐藏态 6/6 与归档一致，其中 5 个调用了工具。`code/check_empty.py:30–49` 记录控制样本、输入及比对方式。6 个原描述控制的最终可见输出均含 `addCriterion`（其中 1 个还含 `scene_classify`），应保留历史异常生成背景。其余归档图像的逐图工具使用次数为 [待确认]，settings 采用 null 与现有证据相符。

## E. Scope and Interpretation: WARN

1. **AGSE 条件特异性。** 通用查询行的数量 A/B 为 47/100、47/100，属性 13/22，关系 10/20，同时存在上述 133/133 单 EOS。它支持“这组历史权重在通用查询条件下的隐藏态读出诊断”。讨论工具增益、推理质量或方法整体优劣时需同时报告该退化现象。原描述缓存补充行为数量 81/100、83/100，属性 16/22，关系 9/20，输入条件差异清晰；其中原描述可携带数词、对象及关系线索（`../第九步_去重与补充/100组工作清单说明.md:12`）。

2. **训练曝光与重复测量。** 全部 133 个图像 ID/RGB 哈希各自唯一，分为 train 96、val 37。数量 78/22，属性 12/10，关系 10/10；数量 A/B 来自同一批 100 图。属性与关系共用 4 张图，数量与另外两维共用 5 张图。属性 22 项对应 11 个反向图对，关系 20 项对应 10 对。区间或检验宜按图像/图对和模板处理相关性；242 项应称为评测项数。数量既有工作清单已处理已发现的三对画面重叠；本轮新增图像仅完成精确 RGB 重复核对，同一地理场景裁剪关系仍为 [待确认]。

3. **文字平衡。** 属性/关系每个句子均在相同配对中出现一次正标签和一次负标签，措辞偏好被显式对冲。数量只确认 A/B 为同义句式；正负数词频率有差异，例如正标签 two 为 44 次、负标签 two 为 31 次，正标签 one 为 20 次、负标签 one 为 21 次。数量部分的高于随机数值需要结合标签分布与图像依赖证据解释。

4. **配对全对现象。** CLIP 的属性 11 对、关系 10 对均恰好“一图正确、一图错误”，双向全对均为 0。CUP_CCB 属性双向全对 2/11、关系 1/10；ASPF 和通用 AGSE 属性均 3/11、关系均 0/10。属性或关系总体接近 50% 时，这项统计能帮助识别固定措辞偏好，建议与单图正确率一起展示。各方法详表见 `independent_evidence.json:pair_both_correct`。

5. **证据规模。** 当前覆盖单一 UCM 数据集、一次确定性生成/评分设置，属性/关系模板较少，包含训练图。支持小样本诊断与后续设计依据；“稳定优于”“普遍提升三类细节”“工具导致提升”等因果或广泛结论仍需补充独立样本与受控实验。

## F. Evaluation Type: WARN — mixed human_eval / synthetic_proxy

该评估使用真实 UCM 图像、自建两句选择任务。数量的目标值来自用户视觉确认，可归为人工标注诊断（human_eval）；属性/关系目标来自助手视觉标注，按本技能“模型生成参考”口径标作 synthetic_proxy，并明确其真实图像基础。原 UCM 数据集的官方 caption/retrieval GT 未直接提供这些逐属性、逐关系标签。评分器输出与标签创建来源已经分开。

## Action Items / Claim Impact

- 可直接报告四种主要方法在当前已锁定评测项上的分数、原始 CLIP 与旧全零提示口径差异、归档先验补充条件，以及生成控制观察。
- 报告 AGSE 通用查询结果时同段注明 133/133 单 EOS、0 次工具调用；报告归档补充时注明首条描述输入、6 图控制复现及异常可见输出。
- 将“133 图、242 项、train/val 分层、人工/助手标注来源”写进结果说明；为属性/关系增加配对双向全对统计。
- 数量 A/B 作为句式稳健性对照分别报告；方法间差值同时说明读出器、LoRA、系统提示及缓存来源差异。
- 属性/关系正式用于论文主结论前，安排独立人工确认；扩展地理场景重复核验与独立测试数据后，再评估可推广性。

本审核未发现分数归一化造高值、缺失结果冒充实测或汇总数值不符的证据。总体 WARN 来自实际生成退化与证据范围；本同系列语义审核保留 provisional 状态。
