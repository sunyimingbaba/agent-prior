import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;raw=r/'原始运行记录'
j=json.loads((r/'analysis.json').read_text());plan=json.loads((r/'plan.json').read_text());v=json.loads((r/'本机独立核验.json').read_text());assert v['ok']
training=j['training'];env=json.loads((r/'preflight.json').read_text());alignment=json.loads((r/'termination_alignment_diagnostic.json').read_text())
model=json.loads((r/'模型本机核验.json').read_text());replay=json.loads((r/'teacher_replay.json').read_text())
names={'historical_legacy':'原 c742＋原流程','c742_strict':'原 c742＋完整标签流程','sft_strict':'训练后＋完整标签流程','sft_legacy_greedy':'训练后＋原流程贪心'}
train=j['groups']['train'];val=j['groups']['val'];m=train['means'];format_ok=j['format_readiness']['passed'];quantity_ok=j['quantity_readiness']['passed']
recommendation='下一节点固定本轮训练后的权重，保留完整工具调用和原系统提示，比较回答闭合标签停止与自然 EOS 停止，重新生成并核对实际末步先验；随后再判断数量奖励训练。'
def protocol_table(split):
    g=j['groups'][split]['means'];lines=['| 相同完整标签控制器 | 采样严格合格率 | 贪心严格合格率 | 采样实际工具调用率 | 采样自产工具结果率 |','|---|---:|---:|---:|---:|']
    for arm in ['c742_strict','sft_strict']:
        s=g[arm]['sample_mean'];gg=g[arm]['greedy'];lines.append(f"| {names[arm]} | {s['strict_quality']['mean']:.2%} | {gg['strict_quality']['mean']:.2%} | {s['tool_usage']['mean']:.2%} | {s['generated_result_failure']['mean']:.2%} |")
    return '\n'.join(lines)
def quantity_table(split):
    g=j['groups'][split]['means'];lines=['| 设置 | 贪心 B PairAcc | 全部采样 B PairAcc | 数量加格式 B PairAcc | 贪心 RR（val） |','|---|---:|---:|---:|---:|']
    for arm in names:
        greedy=g[arm]['greedy'];sample=g[arm].get('sample_mean');selected=g[arm].get('quantity_quality')
        rr=f"{greedy['retrieval_rr']['mean']:.6f}" if 'retrieval_rr' in greedy else '—'
        lines.append(f"| {names[arm]} | {greedy['pairacc_B']['mean']:.2%} | {format(sample['pairacc_B']['mean'],'.2%') if sample else '—'} | {format(selected['pairacc_B']['mean'],'.2%') if selected else '—'} | {rr} |")
    return '\n'.join(lines)
labels={'optimizer_updates_exactly_30':'实际优化器更新恰为 30',
 'strict_sample_quality_at_least_90pct':'train 63 采样严格协议合格率至少 90%',
 'strict_greedy_quality_at_least_95pct':'train 63 贪心严格协议合格率至少 95%',
 'sample_actual_tool_use_at_least_80pct':'train 63 采样实际工具使用率至少 80%',
 'generated_result_failure_at_most_2pct':'train 63 采样自产工具结果率至多 2%',
 'sample_B_pairacc_at_least_matched_control':'train 63 采样 B PairAcc 达到同控制器原模型',
 'legacy_greedy_B_pairacc_drop_at_most_5pp':'原流程贪心 B PairAcc 较 c742 至多下降 5 个百分点'}
checks='\n'.join(f"| {labels[k]} | {'通过' if value else '未达成'} |" for k,value in j['format_readiness']['checks'].items())
gain=train['paired']['sft_strict_minus_c742_strict__sample_mean']['strict_quality'];glo,ghi=gain['interval95']
bgain=train['paired']['sft_strict_minus_c742_strict__sample_mean']['pairacc_B'];blo,bhi=bgain['interval95']
qgain=train['quantity_paired']['sft_strict']['quantity_quality_minus_format_first']['margin_B'];qlo,qhi=qgain['interval95']
text=f'''# 完整工具调用监督训练结论（2026-11-14）

本阶段完成 **15 张图、2 个训练遍次、30 次实际 LoRA 优化更新**，随后完成 935 条固定策略评估轨迹及服务器、本机独立复算。确认 train 63 的严格采样合格率达到 **{m['sft_strict']['sample_mean']['strict_quality']['mean']:.2%}**，实际工具使用率达到 **{m['sft_strict']['sample_mean']['tool_usage']['mean']:.2%}**，贪心严格合格率为 **{m['sft_strict']['greedy']['strict_quality']['mean']:.2%}**。

综合采用条件通过 {sum(j['format_readiness']['checks'].values())}/7 项；差项为同控制器下的数量句对保护指标。数量候选的 7 项准备条件{'通过' if quantity_ok else '尚待达成'}。{recommendation}

## 1. 训练目标和样本职责

上一阶段观察到工具名后直接结束回合的情况，三个校准对照的 180 条采样均未产生工具结束 token。本轮为完整工具调用建立监督路径，并沿用原系统提示、原固定示例与实际图像输入 `Describe the visible scene.`。

训练采用上一轮固定的 15 张 train 图，其余 63 张 train 图用于确认，原 E1 已检查的 val 22 图用于补充诊断。训练图与本轮 85 张评估图的交集为零。继承检查点前一阶段优化的 19 张图与本轮 E1 100 图交集同样为零；原基础模型及冻结 AGSE 的历史来源保留。

每图构造两个助手目标：第一轮为 `<tool_call>scene_classify</tool_call>` 加回合 EOS；第二轮在外部程序真实回填场景结果后生成完整 `<answer>...</answer>` 加 EOS。正文取自之前保存的模型生成回答，按“贪心合格优先，否则首条合格采样”的固定规则选择。正文作为教师上下文，其事实内容随来源保存。

损失计入完整工具目标的 token、回答开始和结束标签中完全落在格式字符范围内的 token，以及 EOS。正文 token、跨越正文边界的混合 token 和提示位置均被屏蔽。一个教师遍次共核对 **{v['training']['format_positions_per_teacher_pass']} 个有效格式位置**、屏蔽 **{v['training']['body_positions_masked']} 个正文/边界位置**；两遍训练完成 30 次更新。格式损失、正文来源、目标 token 与掩码均保存在 `training/teacher_examples.json`。

## 2. 训练设置与实际更新

方案 SHA-256：`{j['plan_sha256']}`。运行前固定的设置与实际记录如下。

| 项目 | 设置或实测 |
|---|---|
| 硬件与框架 | {env['gpu']}；PyTorch {env['packages']['torch']}、Transformers {env['packages']['transformers']}、PEFT {env['packages']['peft']} |
| 参数 | 基础策略 BF16，584 个 LoRA 张量 FP32；仅更新 LoRA；运行时 Dropout p=0 |
| 优化器 | AdamW，学习率 1e-5，betas=(0.9, 0.999)，eps=1e-8，weight decay=0 |
| 每步计算 | 1 张图、两个助手目标分别前后向，按两轮有效格式 token 总数归一化后更新一次 |
| 稳定设置 | 梯度检查点，梯度 L2 范数上限 1，种子 2026111432 |
| 实际更新与模型选择 | 30 次，固定采用第 30 步；第 15 步另存中间状态 |
| 教师格式交叉熵 | {training['initial_teacher_nll']:.9f} → {training['final_teacher_nll']:.9f} |
| 训练阶段耗时 | {training['elapsed_seconds']:.2f} 秒，含教师目标构建、核验和检查点写盘 |
| 训练阶段峰值分配显存 | {training['peak_allocated_bytes']/1024**3:.2f} GiB |

每个优化步骤均记录梯度范数和前后参数哈希；30 次哈希变化及 584 份 Adam 状态的步数均独立核对。初始真实前后向检查保留了非零有限梯度，检查时优化器更新为 0，随后清零再执行正式训练。

![格式训练损失](figures/format_training_loss.png)

初始 LoRA 哈希为 `{training['initial_parameter_hash']}`，训练后为 `{training['final_parameter_hash']}`。最终检查点在评估前固定；确认图与 val 结果均用于固定策略评价。

## 3. 同控制器的协议与实际工具确认

本轮新增三组轨迹：原 c742＋完整标签控制器 425 条、训练后＋相同控制器 425 条、训练后＋原控制器贪心 85 条。另引用已完整核验的原 c742 原流程历史记录。每张主要评估图生成 1 条贪心和 4 条温度 1 采样，种子沿用原 100 图顺序；预算为最多 3 轮、每轮最多 256 token，top-k=0、top-p=1，首 token EOS 屏蔽。所有失败保留。

**确认 train 63：**

{protocol_table('train')}

严格采样合格率的配对变化为 {100*gain['mean']:+.2f} 个百分点，95% 图像 bootstrap 区间 [{100*glo:+.2f}, {100*ghi:+.2f}]。训练后贪心严格合格率为 100%。协议合格依据完整非空 answer、完整调用路径、自产结果标签及长度/轮数终止规则核对；实际工具事件单独重放。

**补充 val 22：**

{protocol_table('val')}

全部区间按图像进行 10,000 次配对 percentile bootstrap，种子 2026111438。val 22 沿用既有诊断角色。

![协议与数量表现](figures/supervision_stability_and_quantity.png)

## 4. 数量句对与完整图库

PairAcc 为冻结 AGSE 对正确数量描述的分数严格高于错误数量描述的比例；A/B 句对及文本向量均与原先逐字节一致。生成输入沿用统一中性文本。数量最大选择使用人工数量标签，在已有四条候选中按 A 间隔选择，随后用 B 写法评价。

**train 63：**

{quantity_table('train')}

同控制器采样 B PairAcc 从 52.38% 变为 39.68%，变化 {100*bgain['mean']:+.2f} 个百分点，95% 区间 [{100*blo:+.2f}, {100*bhi:+.2f}]。原控制器的贪心兼容检查为 73.02% → 69.84%，下降 3.17 个百分点，满足预设的 5 个百分点点估计容差。

训练后的数量加格式选择相对首条合格候选，B 间隔增加 {qgain['mean']:+.7f}，95% 区间 [{qlo:+.7f}, {qhi:+.7f}]；{j['quantity_readiness']['informative_train_images']}/63 张图具有至少两条合格且 A 间隔不同的候选。该组候选诊断保留了可用的跨写法信号。

**val 22：**

{quantity_table('val')}

val 的同控制器采样 B PairAcc 为 72.73% → 77.27%；贪心 RR 为 0.329850 → 0.407181。该组作为补充记录，区间、原流程绝对值及全部选择器结果见 `analysis.json`、`selector_summary.csv` 和 `paired_deltas.csv`。

## 5. 终止状态的探索性对照

同一训练后检查点的两种控制器中，train 有 {alignment['groups']['train']['same_final_decoded_text']}/63、val 有 {alignment['groups']['val']['same_final_decoded_text']}/22 张图生成了完全相同的最终文本和去除 EOS 后的 token 序列。严格流程在回答闭合标签处停止，原流程在 EOS 处停止；两者的工具回填上下文也相差调用结束标签。

以 `124.tif` 为例，严格流程最后发出 `>`，其保存状态来自输入 `answer` 后预测 `>` 的解码步骤；原流程最后发出 `<|im_end|>`，保存状态来自输入 `>` 后预测 EOS 的步骤。两条轨迹使用相同训练后权重并输出相同最终文本，隐藏向量余弦相似度为 0.564122。全部逐图结果见 [终止状态诊断](termination_alignment_diagnostic.json)与 [逐图对照](终止状态逐图对照.csv)。

这些观察提示先验对解码终止位置及工具上下文存在敏感性。下一轮固定权重并保留完整工具调用，将最终回答生成到自然 EOS，再核对预测 EOS 的实际解码步骤；工具上下文与最终停止时点逐项拆开验证。本节为本轮结果后的探索性诊断，运行前固定的采用条件保持原值。

## 6. 采用条件与下一节点

| 运行前固定的检查 | 结果 |
|---|---|
{checks}

综合采用条件：**{'通过' if format_ok else '未通过'}**。本轮的完整调用与回答格式指标达成预设要求，同控制器下的数量保护项仍需处理。数量候选准备条件为 {sum(j['quantity_readiness']['checks'].values())}/7 项；其中有效候选图像要求沿用 60% 比例，按独立确认 63 图固定为 38 张，其余数量阈值保留。

{recommendation} 后续对齐设计见 [下一阶段方案](下一阶段终止状态对齐方案_20261114.md)。

## 7. 独立核验与备份

服务器与本机各核对 30 次更新链、全部教师 token 与掩码、{v['training']['float64_full_vocabulary_probability_rows_checked']} 行完整词表概率、935 条状态与轨迹、{v['generation_rounds_checked']} 轮生成、{v['real_tool_events_checked']} 次真实工具事件、{v['selectors_rebuilt']} 个候选选择结果和 {v['bootstrap_intervals_rebuilt']} 组区间。CPU float64 数量分类 {v['quantity_classifications_equal_float64']}/{v['quantity_classification_checks']}、val 排名 {v['val_ranks_equal_float64']}/{v['val_rank_checks']} 项均与保存值一致。原归档的 503 条历史轨迹同时重新核验。

最终保存检查点重新加载后，30 个教师目标的损失逐项重放，最大绝对差为 {replay['maximum_abs_loss_error']:.9g}。先验缓存始终取最后一次实际解码的最后有效步骤、末层末位置，评分与缓存使用同一张量。文本事实准确性另保留 [255 行人工复核清单](文本事实复核清单.csv)。

模型及优化器完整归档为 `模型检查点/full_tool_supervision_checkpoint_20261114.tar.gz`（{model['archive_bytes']} 字节），584 个 FP32 LoRA 张量及 584 份第 30 步 Adam 状态已本机验证。第 15 步中间状态仍保存在服务器私有目录；最终第 30 步模型、优化器与随机状态已有本机完整副本。

详见 [模型核验](模型本机核验.json)、[服务器核验](服务器独立核验.json)、[本机核验](本机独立核验.json)、[来源核验](final_source_audit.json)与 [复现说明](复现说明_20261114.md)。完整取证归档为 `full_tool_supervision_evidence_20261114.tar.gz`，校验信息见 [备份校验](备份校验.json)。本机原始材料在 `原始运行记录/`，服务器目录为 `{plan['remote_root']}`。
'''
(r/'完整工具调用监督训练结论_20261114.md').write_text(text,encoding='utf-8');print('SFT_REPORT_WRITTEN')
