import argparse
from pathlib import Path
import json
from stage_io import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;raw=r/'原始运行记录'
j=read(r/'analysis.json');plan=read(r/'plan.json');v=read(r/'本机独立核验.json');model=read(r/'模型本机核验.json');assert v['ok'] and model['ok']
tr=j['training'];g=j['groups']['confirmation55'];val=j['groups']['existing_val22'];checks=j['adoption']['checks'];primary=g['macro']['after_minus_before__sample_mean']['pairacc_B'];shared=g['macro']['after_own_minus_shared__greedy']['pairacc_B']
def pp(x):return f"{100*x['mean']:+.2f} 个百分点，95%区间 [{100*x['interval95'][0]:+.2f}, {100*x['interval95'][1]:+.2f}]"
names={'training8':'训练8图','confirmation55':'训练图外确认55图（宏平均52图）','existing_val22':'既有val22图'}
rows=['| 集合 | 数量等权 B：初始 → 第32步 | 宏平均配对变化 | 原始图像等权 B：初始 → 第32步 |','|---|---:|---:|---:|']
for c,label in names.items():
    z=j['groups'][c];before=z['macro_absolute']['before']['sample_mean']['pairacc_B'];after=z['macro_absolute']['after']['sample_mean']['pairacc_B'];change=z['macro']['after_minus_before__sample_mean']['pairacc_B']
    rows.append(f"| {label} | {before:.2%} → {after:.2%} | {pp(change)} | {z['raw']['before']['sample_mean']['pairacc_B']['mean']:.2%} → {z['raw']['after']['sample_mean']['pairacc_B']['mean']:.2%} |")
table='\n'.join(rows)
labels={'optimizer_updates_exactly_32':'实际优化器更新32次','updates_per_count_exactly_8':'数量1、2、3、4各8次更新','updates_per_image_exactly_4':'8张训练图各4次更新',
 'holdout_sample_strict_quality_at_least_90pct':'确认集采样严格合格率至少90%','holdout_greedy_strict_quality_at_least_95pct':'确认集贪心严格合格率至少95%',
 'holdout_sample_actual_tools_at_least_80pct':'确认集采样实际工具率至少80%','holdout_generated_result_failure_at_most_2pct':'确认集采样自产结果率至多2%',
 'holdout_count_macro_B_gain_interval_above_zero':'确认集数量宏平均B增益区间下界为正','holdout_own_minus_shared_macro_B_interval_above_zero':'确认集自身先验较共享先验的宏平均B区间下界为正'}
gate_table='\n'.join(f"| {labels[k]} | {'通过' if value else '未达成'} |" for k,value in checks.items())
if j['adoption']['passed']:
    decision='下一节点采用相同8图与32步预算，增加两个预先固定的训练种子做重复验证，继续核对数量宏平均、共享状态差值和协议质量。'
elif any(not value for key,value in checks.items() if any(x in key for x in ['quality','actual_tools','generated_result'])):
    decision='下一节点先核对协议质量变化，固定本轮失败轨迹，检查数量优势与格式目标的冲突；在此基础上封存格式维持与数量均衡的短程对照设计。'
else:
    decision='下一节点复用本轮保存的全部候选，开展多负例数量奖励的离线诊断：使用同一对象的所有已缓存数量描述构造间隔，再用B写法和共享状态核对候选排序。'
before=g['macro_absolute']['before']['sample_mean']['pairacc_B'];after=g['macro_absolute']['after']['sample_mean']['pairacc_B'];env=read(r/'preflight.json');gate=read(raw/'contract_gate/result.json');settings=read(raw/'training/settings.json');updates=[json.loads(line) for line in (raw/'training/updates.jsonl').read_text().splitlines()];max_lp=max(z['max_logprob_gap'] for u in updates for z in u['probability_replays']);max_mean_lp=max(z['mean_abs_logprob_gap'] for u in updates for z in u['probability_replays']);min_cos=min(z['final_teacher_state_cosine'] for z in gate['probability_rows'] if z['final_teacher_state_cosine'] is not None)
text=f'''# A100数量均衡32步短训结论（2026-11-15）

本阶段完成 **8张图、32次正式优化器更新、128条训练采样及425条训练后评估轨迹**。每种数量1–4各8次更新、每图4次更新。训练图外确认集的采样数量宏平均B PairAcc为 **{before:.2%} → {after:.2%}**，配对变化为 **{pp(primary)}**。正式采用条件通过 **{sum(checks.values())}/9** 项。

确认集训练后贪心自身先验相对固定共享先验的数量宏平均B变化为 **{pp(shared)}**。{decision}

## 1. 样本职责、初始模型和运行环境

数量1、2、3、4各选2张已确认train图，按上一阶段固定顺序训练。图像为 {', '.join(plan['training_images'])}。其余55张train图用于本轮训练图之外的确认，既有val22图单列补充；确认集内数量1–4共52图，每种数量在宏平均中占四分之一。数量6/7另列原始图像等权结果和分组表。所有输入、轨迹与失败样本保留。

初始策略为上一轮固定SFT第30步检查点，LoRA张量哈希 `{tr['initial_parameter_hash']}`。本轮模型固定取第32次正式更新，最终哈希 `{tr['final_parameter_hash']}`。方案SHA-256为 `{j['plan_sha256']}`。

服务器实际为 {env['gpu']}，81920MiB显存；PyTorch {env['packages']['torch']}、TorchVision {env['packages']['torchvision']}、Transformers {env['packages']['transformers']}、PEFT {env['packages']['peft']}。沿用原A100环境及 `/private` GPFS挂载。此前5090适配尝试按用户指示切回A100，相关日志另行保留。

## 2. 概率、状态与固定基线核对

正式更新前重放 `102.tif` 的贪心与首条采样，生成token及末步隐藏状态与已归档基线逐项一致；425条冻结AGSE评分重放最大误差为0，val排名一致。本轮沿用这一固定基线，训练后采用同一输入、同一种子和同一控制器。

生成采用原系统提示和示例、统一输入 `Describe the visible scene.`、真实 `scene_classify` 回填、完整工具关闭标签及回答原生EOS。采样温度1、top_k=0、top_p=1、min_new_tokens=1，最多3轮、每轮256 token。先验为实际generate最后有效解码步的末层末位置BF16[3584]；同一张量保存并传入冻结评分器。

逐token核对了首token EOS屏蔽、助手目标位置和17个额外padding的裁剪。实际解码与完整前缀重算之间，训练token的最大logprob差为{max_lp:.6f} nat，最大轮平均绝对差为{max_mean_lp:.6f} nat；运行前阈值分别为0.25和0.05 nat。预检中末步状态余弦最低为{min_cos:.9f}，阈值为0.999。先验缓存与评分输入使用同一实际生成张量。正式更新前的临时Adam单步内存验证已恢复初始LoRA哈希，记录在 `contract_gate/`，其正式策略更新数为0。该检查与本轮32次正式更新分别登记。

## 3. 实际训练设置与执行

每次对同一训练图采样4条实际轨迹，奖励为质量合格时的0.2加冻结下游AGSE的A数量句对间隔。组内奖励按样本标准差标准化，加1e-8稳定项；对各轨迹实际助手token的logprob和进行组平均策略梯度更新，工具回填和用户提示作为上下文。

| 项目 | 实际设置或记录 |
|---|---|
| 参数 | BF16基座、{settings['trainable_tensors']}个FP32 LoRA张量；{settings['trainable_parameters']}个可训练参数 |
| 优化器 | AdamW，学习率5e-6，betas=(0.9,0.999)，eps=1e-8，weight decay=0.01 |
| 稳定措施 | 梯度检查点、Dropout p=0、梯度L2范数上限1、逐轨迹逐轮反向 |
| 更新预算 | 32次；每图4次，每数量8次；中间第16步与最终第32步保存 |
| 训练采样 | {tr['quality_samples']}/{tr['samples']}条通过质量检查；真实工具事件{tr['actual_tool_events']}次 |
| 训练阶段耗时 | {tr['elapsed_seconds']:.2f}秒，含模型加载与检查点保存 |
| PyTorch峰值分配显存 | {tr['peak_allocated_bytes']/1024**3:.2f}GiB |
| 优化器核验 | {model['optimizer']['parameter_states']}份Adam状态，步数最小/最大均为32 |

每条训练采样均保存生成概率、真实工具、状态、A/B分数与奖励。A用于奖励，B用于评价；32次更新的参数哈希链、梯度范数和概率重算记录均保留。

## 4. 数量均衡与共享状态结果

下表以每图4条采样平均B PairAcc汇总，数量宏平均对1、2、3、4各赋相同权重：

{table}

![数量宏平均与共享先验对照](figures/balanced32_macro_and_shared_control.png)

共享先验控制固定参考图为 `102.tif`，分别使用初始和第32步模型的贪心状态广播到全部目标图；自身先验使用各目标图的贪心状态。确认集训练后的自身/共享数量宏平均B PairAcc为 {g['macro_absolute']['after']['greedy']['pairacc_B']:.2%}/{g['macro_absolute']['after']['shared']['pairacc_B']:.2%}，差值区间见本报告开头。

确认集训练后采样严格合格率为 {g['raw']['after']['sample_mean']['strict_quality']['mean']:.2%}，实际工具使用率为 {g['raw']['after']['sample_mean']['tool_usage']['mean']:.2%}，贪心严格合格率为 {g['raw']['after']['greedy']['strict_quality']['mean']:.2%}。

既有val22的贪心RR为 {val['raw']['before']['greedy']['retrieval_rr']['mean']:.6f} → {val['raw']['after']['greedy']['retrieval_rr']['mean']:.6f}。RR沿用原391条描述、原正例集合与严格大于最佳正例分数的零基排名规则。原始图像等权、数量分组及A指标均保存在 `analysis.json` 和CSV中。

所有区间按图像重采样10,000次，数量宏平均在各数量内分层重采样，主种子2026111555。本轮仅有一个正式训练种子，区间描述当前图像集合的配对变化。

## 5. 预先固定的采用条件与下一节点

| 条件 | 结果 |
|---|---|
{gate_table}

{decision} 第32步模型、优化器、随机状态和本轮所有失败轨迹均已保存。

## 6. 独立核验与备份

服务器和本机分别核对32次更新、128条训练采样、850条初始/训练后自身先验记录、170条共享控制，共1148条状态与评分记录。训练概率token共核对{v['training_probability_tokens_checked']}个，完整词表float64概率见证{v['float64_full_vocabulary_witness_rows']}行；float64数量判别一致{v['float64_quantity_classifications_equal']}/{v['quantity_classification_checks']}，val排名一致{v['float64_val_ranks_equal']}/{v['val_rank_checks']}。另重建{v['per_image_rows_rebuilt']}行逐图结果与{v['bootstrap_intervals_rebuilt']}组区间，并递归核对前序EOS、SFT和分歧诊断证据。

最终LoRA、优化器和随机状态完整归档为 `模型检查点/balanced_quantity32_checkpoint_20261115.tar.gz`，{model['archive_bytes']}字节，SHA-256 `{model['archive_sha256']}`；16份文件与584个FP32适配张量已本机核对。证据包为 `balanced_quantity32_evidence_20261115.tar.gz`，校验见 [备份校验](备份校验.json)。

复现步骤见 [复现说明](复现说明_20261115.md)，原始记录在 `原始运行记录/`；[425条训练后文本事实清单](训练后文本事实复核清单.csv)保留人工事实准确性 `[待确认]`。服务器目录为 `{plan['remote_root']}`。
'''
(r/'A100数量均衡32步短训结论_20261115.md').write_text(text,encoding='utf-8')
(r/'下一节点.md').write_text('# 下一节点\n\n'+decision+'\n',encoding='utf-8');print('A100_BALANCED32_REPORT_WRITTEN')
