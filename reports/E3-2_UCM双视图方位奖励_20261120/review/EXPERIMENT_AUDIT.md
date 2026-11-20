# Experiment Audit Report

**日期**：2026-11-20  
**审阅者**：独立新建 Codex 审阅代理 `/root/paired_rotation_integrity`（调度记录：gpt-6-astra、ultra）；本地只读 CPU 核对。  
**项目**：UCM 双视图方位奖励，两个采样种子、三种配方。  
**Overall verdict / Integrity status**：**WARN / warn**。  
**review_independence**：same-family；**acceptance_status**：provisional。  
**审阅正文 SHA-256**：`db6fba2d68c80a4a880991a342cc0acd0693cac86fe3349646bd9fe5073e58f6`。

实验数值、训练与推理记录、检查点及最终部署决策互相吻合。“保留历史权重”的结论获得本轮证据支持。WARN 来自已披露的助手像素标签和探索性评估范围，当前数值核验通过。审阅期间要求补充的标签来源、每图首句条件和召回统计单位已写入以上哈希对应的正文。

## A. Ground Truth Provenance：WARN

普通检索的正例来自 UCM 原始图文归属。验证表含 210 图；391 个词元去重文本向量通过 `caption_order` 展开为 1,050 条描述，重复描述仍有原图归属。评分用这一归属判断命中。方向题由既有像素标注与同模型家族助手复核形成，E1 数量沿用用户确认，属性和真实关系沿用助手像素复核；这些标签的出处在主文第 7、47 行明确。方向/E1 诊断采用人工构造的正反描述，属于带像素依据的代理标签评价。参考答案沿用事先冻结题库，评分阶段读取标签与向量计算分差。

核对了三张训练候选面板及前轮三张方向面板；保留 16 张训练图覆盖 5 个场景块，方向验证 16 图覆盖 10 个块。训练与完整验证按文件名、RGB 指纹均零交集，方向图与 E1 图也零交集。180° 旋转交换正确/反向分差的符号；90° 顺转以 PIL `ROTATE_270` 实现，“上→右→下→左”的谓词变换通过独立像素坐标检查，64 条 C/D 正反句保持同一对象与参照。实际推理记录均匹配源图指纹和预定朝向。

证据：`code/shared.py:23`，`code/evaluate.py:61`、`:103`、`:110`，`code/analyze.py:69`、`:119`，`code/prepare_quarter.py:5`，`inputs/lock.json:1`，`results/source_pixels.json:1`，`实验交付.md:7`、`:47`。源图原始 Arrow 的独立逐像素读取在私有服务器执行；本审阅使用本地面板、冻结清单及落盘指纹核对其一致性。

## B. Score Normalization：PASS

验证指标直接使用单位范数图文向量的余弦分数、正确描述减反向描述的原始分差、固定题数下的严格正分比例，以及六项召回均值。原始分差和逐源图结果均已保存。模型自身的最值、均值和方差只用于训练候选的组内名次与优势标准化；正式验证分母为预定图数、题数或查询数。

匹配的 `joint` 与 `separate` 共用每视图 4 条采样、固定槽位配对、采样种子、逐视图排名/样本标准差和每轨迹 1/8 梯度权重。`joint` 将方向项替换为两个正确有符号分差的较小值。两种子首组更新前的轨迹和原始分数完全相同，首组奖励与更新后参数开始分化。单视图配方的 8 条统一标准化及曝光差异已在主文第 9 行说明。

证据：`code/rewards.py:6`，`code/train.py:92`、`:101`、`:111`、`:118`，`code/metrics.py:8`、`:15`，`results/training_audit.json:1`，`实验交付.md:9`。

## C. Result File Existence and Numerical Replay：PASS

本审阅直接读取原始记录与张量，在 CPU 上采用独立排序和分差计算复核：

| 项目 | 独立核对结果 |
| --- | --- |
| 13 个主状态 | 6,292 条记录的来源、查询、种子、朝向、隐藏态哈希及向量绑定通过 |
| 普通检索 | 32,760 个按源图汇总的召回分项最大差 0；主表 35 个数值格一致 |
| 主方向/训练图方向 | 1,664 个有符号分差，最大绝对差 1.166×10⁻¹⁵，判定一致 |
| E1 | 3,016 个分差，最大绝对差 1.111×10⁻¹⁵，判定一致 |
| 配对区间 | 396 项主比较、36 项额外角度、180 项 E1 比较复算最大差 0 |
| 额外角度 | 7 个状态、224 条记录，双角度及四角度交集计数一致 |
| 先验交叉 | 896 个文本分差格；自然两格向量逐值等于部署输出，汇总差 ≤5.56×10⁻¹⁷ |
| 训练执行 | 96 个候选组、768 条轨迹、7,535 个生成 token、94 个非零梯度组；优势重算差 0 |
| 模型归档 | 12 个完整检查点、6 个末步恢复文件、36 个无损差分块的 SHA-256 全部吻合 |
| 历史起点 | 前轮 452 条 native/generic/detail 记录及 hidden/visual 张量逐项一致 |
| 双端凭据 | 25 项本机/服务器凭据与当前规范输出哈希一致；24 项逐字节相同 |

训练恢复状态中的日志、初始参数指纹、代码/输入锁、584 个活动张量及优化器步数互相匹配。对数概率重放最大误差 0 来自实际训练保存的逐步核验值；本审阅独立检查了所有保存 token/logp 的长度、有限性和奖励/优势，GPU 概率重算沿用执行记录。早期 Lightning 回调加载异常已完整保留，完成标记及后续六组轨迹均齐全。

较弱侧末步原首句 mR 为 A 58.341779%、B 59.662652%，历史为 60.055408%；双视图分别 3/32、4/32，匹配分别计分组为 3/32、0/32。`decision.json` 的四项逐种子判定和 `promote=false` 全部吻合。四角度连续正确为两种较弱侧权重均 0/32。双侧训练正分候选为 1、4、4、4/64；任意重配的最大总数相同。16 图的采样覆盖为双侧 6、仅原侧 6、仅旋转侧 4。

证据：`results/states.json:1`，`results/decision.json:1`，`results/paired_contrasts.json:1`，`results/quarter_stats.json:1`，`results/prior_swap.json:1`，`results/training_audit.json:1`，`results/weight_restoration.json:1`，各 `eval/*/complete.json`、`policy_*/steps.json`、`policy_*/resume.pt`；具体哈希在配套 JSON。

## D. Dead Code / Execution：PASS

训练、主推理、额外角度和交叉读出均有独立输出与完成标记。`fixed`、`recalls`、`ci`、方向 `details`、训练 `calculate` 以及四格分解都沿实际流水线调用，其结果可由已保存向量和轨迹重新取得。两个评估脚本版本的来源哈希分别对应当前 `evaluate.py` 与 `review/initial_evaluator.txt`；差异为额外角度分支，原四个主分块的输入和生成/读出路径保持一致。

证据：`code/train_queue.sh:7`，`code/eval_queue.sh:17`，`code/post_queue.sh:6`，`code/quarter_queue.sh:7`，`code/swap_queue.sh:5`，`code/analyze.py:100`、`:107`、`:174`，`code/analyze_swap.py:27`，`logs/*_complete`。

## E. Scope Assessment：WARN

实际规模为 UCM 16 张训练图、两个采样种子、三配方，完整验证 210 图；方向题为重复使用的 16 源图、两措辞。32 项双视图题的抽样单位为 16 张源图。置信区间固定已经运行的两个种子和图库，对同源措辞/视图整体重采样。额外 90° 评价继续使用同一批图；先验交换在中间结果后登记，按探索性机制诊断解释。训练候选覆盖描述本轮已采样轨迹。

主文已明确这些条件、单视图配方的并行变化、末步固定规则及正式测试集的后续用途。依据当前证据，继续保留历史权重和优先补充训练方向图对的决策成立。后续若形成一般性方位能力或跨种子稳健性的论文结论，应加入独立人工复核的新增方向图及更多种子。

证据：`inputs/protocol.json:20`，`inputs/quarter_protocol.json:11`，`inputs/prior_swap_protocol.json:2`，`code/metrics.py:23`，`code/analyze.py:150`，`实验交付.md:25`、`:49`、`:51`、`:55`。

## F. Evaluation Type Classification：WARN / mixed

- **real_gt**：UCM 原图文归属下的 native 与 generic 检索；native 为每图自带第一条目标描述条件下的历史部署口径。
- **synthetic_proxy（pixel-grounded assistant labels）**：训练方向、验证 C/D 与额外旋转方向，及 E1 属性/关系正反句；标注与像素复核来源已披露。
- **human_eval-derived labels**：E1 数量题沿用既有用户确认，随后由冻结分差自动计分。
- **self_supervised_proxy / intervention diagnostic**：同源两先验×两像素的冻结读出交叉响应及代数分解；这是已登记的探索性机制对照。

证据：`实验交付.md:7`、`:47`、`:49`，`inputs/e1_extension.json:2`，`inputs/prior_swap_protocol.json:5`，`code/analyze_swap.py:23`。

## Claim Impact and Action Items

1. **支持**：两个较弱侧末步权重均未达到预设晋级条件，继续使用历史部署权重。
2. **支持，保留当前条件说明**：种子 B 相对匹配对照新增 4 项双视图正确，种子 A 得失各 3 项；检索变化方向随种子改变。
3. **支持，探索性范围**：先验互换会改变固定读出器的方向排序；两种较弱侧权重在四角度交集均为 0/32。
4. **后续工作**：保留当前标签来源和重复验证面板说明；新增独立方向图并开展人工复核后，再评估泛化与稳健性。

本审阅只写入两份审计文件，未变更训练、评估、原始结果或主报告。调度及沟通 trace：`.aris/traces/experiment-audit/2026-11-20_rotation/`。
