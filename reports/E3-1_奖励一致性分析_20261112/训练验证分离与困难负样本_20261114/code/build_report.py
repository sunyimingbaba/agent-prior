import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;raw=r/'原始运行记录'
j=json.loads((r/'analysis.json').read_text());v=json.loads((r/'本机独立核验.json').read_text());assert v['ok']
audit=json.loads((r/'split_audit.json').read_text());plan=json.loads((r/'实验方案.json').read_text())
branch=j['pilot'];means=j['policy_means'];names={'baseline':'初始 LoRA','format_only':'仅格式奖励','global_margin':'训练库间隔＋格式奖励','scene_margin':'同场景间隔＋格式奖励'}
table=['| 模型点 | 实际更新 | 贪心平均 RR | 贪心合格率 | 采样平均 RR | 采样合格率 |','|---|---:|---:|---:|---:|---:|']
for p in j['policies']:
    m=means[p];updates='—' if p=='baseline' else j['training'][p]['optimizer_updates']
    table.append(f"| {names[p]} | {updates} | {m['greedy']['utility_rr']['mean']:.6f} | {m['greedy']['quality']['mean']:.2%} | {m['sample_mean']['utility_rr']['mean']:.6f} | {m['sample_mean']['quality']['mean']:.2%} |")
paired=['| 配对比较 | 平均 RR 差 | 95% 图像 bootstrap 区间 | 胜/平/负 |','|---|---:|---:|---:|']
for key,modes in j['paired'].items():
    policy,base=key.split('_minus_')
    for mode,label in [('greedy','贪心'),('sample_mean','采样均值')]:
        s=modes[mode]['utility_rr'];lo,hi=s['interval95']
        paired.append(f"| {label}：{names[policy]} − {names[base]} | {s['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {s['wins']}/{s['ties']}/{s['losses']} |")
training=['| 分支 | 尝试组数 | 实际更新 | 合格轨迹 | 真实工具执行 |','|---|---:|---:|---:|---:|']
for arm,t in j['training'].items():training.append(f"| {names[arm]} | {t['attempted_steps']} | {t['optimizer_updates']} | {t['quality_pass_samples']}/{t['total_samples']} | {t['real_tool_executions']} |")
decision='达到' if j['decision']['expand_criterion_passed'] else '未达到'
next_action='下一节点先开展独立训练种子复核，再决定扩大预算。' if j['decision']['expand_criterion_passed'] else '本轮暂缓扩训。下一节点优先检查候选负例的语义有效性，利用已有人工核验的数量图文对核对训练信号，再确定后续奖励方案。'
branch_text='同场景分支达到预设额外信号阈值，已加入正式训练。' if branch['scene_arm_enabled'] else '同场景分支低于预设额外信号阈值，正式训练保留格式与全训练库间隔两个分支。'
chosen=j['decision']['chosen_by_training_only_gate'];primary=means[chosen]['greedy']['utility_rr']['mean'];baseline=means['baseline']['greedy']['utility_rr']['mean'];fmt=means['format_only']['greedy']['utility_rr']['mean']
labels={'greedy_rr_ci_above_baseline':'贪心 RR 相对初始模型的区间下界 > 0','greedy_rr_ci_above_format_only':'贪心 RR 相对格式组的区间下界 > 0','greedy_rr_ci_above_global_margin':'贪心 RR 相对全训练库间隔组的区间下界 > 0','greedy_rr_at_least_initial':'贪心平均 RR 至少达到初始模型','greedy_quality_within_5pp':'贪心合格率距初始模型至多下降 5 个百分点','optimizer_updated':'已发生优化器更新'}
checks='\n'.join(f"| {labels[key]} | {'通过' if value else '未达成'} |" for key,value in j['decision']['checks'].items())
comparison=j['paired']['global_margin_minus_format_only']['greedy']['utility_rr']
sampling=j['paired']['global_margin_minus_format_only']['sample_mean']['utility_rr']
sample_quality=j['paired']['global_margin_minus_format_only']['sample_mean']['quality']
interpretation='配对区间跨过 0，当前图像集合下的额外检索收益仍需复核。' if comparison['interval95'][0]<=0<=comparison['interval95'][1] else '配对区间位于 0 的一侧，方向与幅度按上表记录。'
run=json.loads((r/'run_result.json').read_text());replay=json.loads((raw/'replay_verification/result.json').read_text())
omitted='、'.join(k for k,x in audit['class_coverage'].items() if x['eligible_images']==0)
text=f'''# 训练验证分离与困难负样本对照（2026-11-14）

本阶段完成训练图像、训练输入描述和奖励文本库与验证数据的分离，完成同场景候选负例对照，并在新 50 图上比较本轮实际训练的模型。**预设扩训条件{decision}。** {next_action}

## 1. 数据分离结果

原 train Arrow 有 1,680 张、val Arrow 有 210 张，两者图像二进制哈希交集为零。原训练描述共有 1,637 条唯一原始字符串；按 Unicode NFKC、大小写归一、合并空白和去掉末尾空格/句末标点进行规范化后，剔除与全部 val 描述相同的训练文本，得到 **1,437 条训练奖励描述**，与验证描述的规范化文本交集为零。

1,277 张训练图像保留有效正例和同场景候选负例。按种子 2026111403 从 19 个有资格的场景各选一图，固定 19 组训练图像。{omitted} 两类的训练描述在本轮去重后均为空，类别覆盖如实记录在 [split_audit.json](split_audit.json)。每张训练图使用第一条保留描述作为生成输入，正例使用该图全部保留的有效描述。

新增验证 50 图以种子 2026111404 从剩余 118 张 val 图中随机抽取，排除了此前 20 张优化图、旧 22 图和上一阶段新 50 图，共 92 张。验证先验继续按每幅图像及其第一条原始描述生成。验证排名图库沿用冻结的 391 条唯一 val 描述及原始文本特征；训练过程加载训练图像与训练文本特征。

同场景标签来自原 `train_scene_cls.py` 的 `(文件名编号−1)//100` 规则，并与已保存场景头的类别表逐项核对。候选负例按训练标注配对关系与这一场景标签建立，完整排除目标图全部保留的有效描述。其细粒度语义互斥性保留为后续人工核验项。

初始 LoRA 和冻结最终 AGSE 检索器沿用既有检查点；历史来源按既有资产记录保留。本轮落实新优化图像、输入描述、奖励文本和验证数据的分离。测试 Arrow 保持原文件，本阶段使用 train 与 val。

方案 SHA-256：`{j['plan_sha256']}`，见 [实验方案](实验方案.json)。划分、图库、提示、随机种子、条件分支和扩训标准均在候选对照及本轮验证生成前固定。

## 2. 同场景困难负例的候选级对照

初始策略在 19 张训练图上每图采样 4 条，在真实工具流程下生成 76 条候选轨迹，分别计算全训练库最难负例间隔和同场景最难负例间隔。二者使用相同的正确描述、冻结检索器与格式奖励。

两种间隔的区别在于负例集合：全训练库从所有 1,437 条训练描述中排除有效正例后取最大余弦，同场景版本将这一候选集合进一步限制到同一场景。全训练库中的最难负例已有 **{branch['global_negative_same_scene_count']}/76** 条来自同场景。

预设条件要求至少 5/19 个图像组的两种总奖励在标准化组内优势上出现最大绝对差 > 0.05。实际为 **{branch['actual_groups']}/19 组**。{branch_text} 决策完全使用训练图像候选，未使用新 50 图的评分。完整逐组数值保存在原始记录 `pilot/branch_decision.json`。

### 探索性负例语义核查

固定抽取每图第 0 条候选，对 19 张训练图和对应负描述进行助手视觉初核。全训练库负例中，10/19 条疑似仍与图像兼容，6 条有可见冲突，3 条待确认；同场景负例分别为 11/19、4/19、4/19。该核查作为解释性诊断，保留原标注与全部训练结果。

例如，1209.tif 的正负描述均表达植被茂盛的中等密度住宅区，331.tif 的负描述为可见的海滩与海浪；1727.tif 的 R/L 和 2049.tif 的三/四块网球场则具有明确可辨差异。这提示后续需要逐项核实负描述与图像的语义冲突性。视觉初核有待独立人工复核，记录见 [负例核查](negative_review/assistant_visual_review.csv)，对应 5 页图文联查表位于同目录。

## 3. 相同预算的短程训练

{'\n'.join(training)}

每个分支均从同一原始 LoRA 重新加载，初始参数哈希为 `{plan['inherited_initial_hash']}`。每分支 19 次图像组尝试，每组 4 条，学习率 5e-6，AdamW weight decay 0.01，梯度裁剪 1.0。组内样本标准差 ≤ 1e-8 时按方案跳过更新。

全训练库间隔奖励为 `0.2 × quality_usable + best_positive_cosine − hardest_training_negative_cosine`。格式对照仅保留前一项；两个分支均保存全部间隔以便重放。

基座 BF16、适配器 FP32、运行时 dropout 为 0。真实工具调用、消息分离、EOS/填充屏蔽、assistant 有效 token log probability 和最终解码步骤状态均沿用已验证共享实现。独立单步检查完成一次有限梯度更新，该项检查记录单独保存；正式分支分别从相同初始模型开始。

## 4. 新 50 图主结果

{'\n'.join(table)}

采样每图使用种子 0、1 的两条轨迹，先在图内取均值再在图像间等权平均；全部失败轨迹保留，质量合格率单列。三个模型点在同一组本轮新 50 图上比较。主分析指定模型的贪心 RR 为 {primary:.6f}，初始模型为 {baseline:.6f}，格式对照为 {fmt:.6f}。

{'\n'.join(paired)}

采样条件下，间隔组相对格式组的平均 RR 差为 {sampling['mean']:+.6f}，95% 区间 [{sampling['interval95'][0]:+.6f}, {sampling['interval95'][1]:+.6f}]；两者合格率为 {means['global_margin']['sample_mean']['quality']['mean']:.0%} 与 {means['format_only']['sample_mean']['quality']['mean']:.0%}，差 {100*sample_quality['mean']:+.0f} 个百分点，95% 区间 [{100*sample_quality['interval95'][0]:+.0f}, {100*sample_quality['interval95'][1]:+.0f}] 个百分点。真实工具调用比例分别为 {means['global_margin']['sample_mean']['tool_usage']['mean']:.0%} 与 {means['format_only']['sample_mean']['tool_usage']['mean']:.0%}。这些次要结果提示后续奖励设计还需关注采样时的输出与工具流程稳定性。

全训练库间隔组相对格式组的贪心平均 RR 差为 {comparison['mean']:+.6f}，50 图中有 {comparison['ties']} 图排名指标相同。{interpretation} 回答质量和检索指标分开保存，便于核对格式稳定性的贡献。

配对区间使用 10,000 次图像 bootstrap，种子 2026111406，2.5/97.5 百分位。区间衡量当前一次训练下的图像抽样变化；训练随机性的变化留待独立种子复核。余弦间隔、工具调用率及逐图数据见 [配对数据](paired_policy_deltas.csv)、[逐图数据](per_image_policy.csv)、[完整评分](evaluation_scores.csv)。

![分离后的奖励对照](figures/separated_reward_fresh50.png)

图左为三个模型点的贪心 RR，中间为回答合格率，右侧为按训练候选条件预先确定的主要模型相对对照的 RR 差及区间。

## 5. 扩训判断

本轮主要模型由训练候选条件固定为 `{chosen}`。对应的预设检查如下。

| 机器检查 | 结果 |
|---|---|
{checks}

总体：**{decision}预设扩训条件。** {next_action} 本轮结果与前一轮采用不同训练图像、训练提示和新验证子集，后续分析以各轮内部的配对对照为依据。

## 6. 核验、备份与复现

服务器及本机核对了 {v['states_and_traces_checked']} 条状态与轨迹、{v['rounds_checked']} 轮生成、{v['real_tool_events_checked']} 次真实工具事件、{v['training_and_pilot_rewards_checked']} 条训练/候选/单步检查奖励、{v['evaluation_scores_checked']} 条验证评分和 {v['paired_bootstrap_intervals_checked']} 组配对区间。所有训练正例和同场景候选负例集合均从元数据重建，图像、规范化描述、提示与新 50 图选择规则分别复核。

真实工具在训练与验证共 {replay['real_tool_images_recomputed']} 幅图像上重跑，逐事件核对。训练奖励重放最大间隔误差为 {replay['max_margin_replay_error']:.9g}。本机 CPU float64 重算排名 {v['rank64_equal_count']}/{v['total_rank_checks']} 条一致；全部保存 FP32 相似度下的奖励与排名复核通过。训练分支条件同样独立重建，见 [服务器核验](服务器独立核验.json)、[本机核验](本机独立核验.json)。

模型归档逐文件核对，并从每份 584 个有限 FP32 LoRA 张量重建训练时的参数哈希，见 [模型备份清单](模型备份清单.json)。原始代码、环境兼容文件、模型和 train/val Arrow 在运行后核对原有哈希，见 [来源核验](final_source_audit.json)。

服务器沿用 A100-SXM4-80GB、PyTorch 2.4.1+cu124、Transformers 4.57.2、PEFT 0.10.0、NumPy 1.26.4。本轮主训练及验证流程耗时 {run['elapsed_seconds']/60:.2f} 分钟，各阶段退出码均为 0。

原始记录保存在本机 `原始运行记录/` 及服务器 `/private/AGSE_E3_1_rebuild_20261112/separated_hard_negative_20261114`；模型独立保存在 `模型检查点/`。结果取证归档为 `separated_reward_evidence_20261114.tar.gz`，见 [备份校验](备份校验.json)。复现流程见 [复现说明](复现说明.md)。
'''
(r/'训练验证分离与困难负样本结论_20261114.md').write_text(text)
print('REPORT_WRITTEN')
