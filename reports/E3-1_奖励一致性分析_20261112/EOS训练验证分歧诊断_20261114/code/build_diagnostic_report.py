import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
def read(n):return json.loads((r/n).read_text(encoding='utf-8'))
j=read('analysis.json');plan=read('plan.json');v=read('本机独立核验.json');assert v['ok'];cases=read('inputs/evaluation_cases.json')
def group(split,count):return next(x for x in j['stratified'] if x['selector']=='sample_mean' and x['split']==split and x['dimension']=='actual_count' and x['group']==str(count))
def macro(split):return next(x for x in j['count_standardized'] if x['selector']=='sample_mean' and x['split']==split)
def control(arm,split):return next(x for x in j['control_summary'] if x['arm']==arm and x['split']==split)
rows=['| 实际数量 | 确认 train 图数 | train 标签停止 → EOS | 既有 val 图数 | val 标签停止 → EOS |','|---|---:|---:|---:|---:|']
for count in [1,2,3,4]:
    t=group('train',count);z=group('val',count);rows.append(f"| {count} | {t['images']} | {t['before_pairacc_B']:.2%} → {t['after_pairacc_B']:.2%} | {z['images']} | {z['before_pairacc_B']:.2%} → {z['after_pairacc_B']:.2%} |")
count_table='\n'.join(rows)
rows=['| 固定 AGSE 的先验输入 | train 63 B PairAcc | val 22 B PairAcc | val RR |','|---|---:|---:|---:|']
labels={'own_answer_close':'各图自身标签停止状态','own_eos':'各图自身 EOS 状态','shared_answer_close':'同一参考图标签停止状态广播','shared_eos':'同一参考图 EOS 状态广播','zero_input':'3584 维零向量（保留投影层偏置）','historical_cache':'历史训练缓存（原始首条描述条件）'}
for arm,label in labels.items():
    t=control(arm,'train');z=control(arm,'val');rows.append(f"| {label} | {t['pairacc_B']:.2%} | {z['pairacc_B']:.2%} | {z['retrieval_rr']:.6f} |")
control_table='\n'.join(rows);d=next(x for x in j['decomposition'] if x['selector']=='sample_mean');mt=macro('train');mv=macro('val')
pilot=[]
for c in [1,2,3,4]:pilot.extend([x['image'] for x in cases if x['split']=='train' and x['actual_count']==c][:2])
assert len(set(pilot))==8
pilot_plan={'role':'prospective next-stage design; not executed','training_images':pilot,'selection':'first two inherited confirmation-train cases per actual count 1/2/3/4, fixed inherited order',
 'training_images_per_count':2,'planned_actual_optimizer_updates':32,'planned_updates_per_count':8,'planned_updates_per_image':4,
 'confirmation_train_holdout_images':[x['image'] for x in cases if x['split']=='train' and x['image'] not in pilot],
 'supplementary_existing_val_images':[x['image'] for x in cases if x['split']=='val'],'starting_policy':'fixed SFT update-30 LoRA 10cb14e3f10f94eeaa8cc98f84b5fe7f9abd05d97cbc26ad4b219efde88017cb',
 'rollout':'same complete real-tool calls, neutral user prompt, temperature 1, native EOS, actual last decoding-step prior',
 'reward':'inherited A quantity margin and existing quality bonus',
 'evaluation':'B and original391-caption RR; image-unit paired intervals; equal true-count weights for 1..4, with all-count raw metrics retained',
 'new_checks':['training/inference token-probability and selected state agreement before update 1','fixed update-32 checkpoint','confirmation-train count-macro B change interval lower above zero','own-state advantage versus the same fixed-reference shared-state control reported with interval','format and real-tool-use retained'],
 'checkpoint_selection':'fixed final update; existing val supplementary, excluded from checkpoint choice','stage_status':'plan only; training code and complete numerical gates to be sealed before launch'}
(r/'下一阶段数量均衡试验方案.json').write_text(json.dumps(pilot_plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=f'''# EOS 训练与验证分歧诊断（2026-11-14）

本阶段完成 **425 条既有轨迹的逐图诊断、340 次新增冻结模型对照及170次原评分重放**，新增生成和优化器更新均为 **0**。85 张图的缓存行号、原始首条描述与先验哈希均对应；实际 Dataset 读取还核对了 {j['scoring']['dataset_caption_samples_checked']} 个描述样本。

1. EOS 终止后，下游 AGSE 对“二”这一数量的偏好明显增强；相同最终文本对应的状态位置变化产生了评分变化。
2. train 与 val 的数量组成差异影响了总准确率：train 以双目标图为主，val 以单目标图为主。按数量 1–4 等权后，采样 B PairAcc 变化分别为 **{100*mt['delta_pairacc_B']:+.2f} 个百分点**和 **{100*mv['delta_pairacc_B']:+.2f} 个百分点**，两组区间均跨过 0。
3. 将一张参考图的同一状态用于全部目标图，也重现了 train 上升、val 下降。下一节点采用数量均衡的短程训练设计，并以数量宏平均和共享状态对照核对逐图收益。

## 1. 逐图映射和模型来源

本轮复用上一阶段固定的 63 张确认 train 图和 22 张既有 val 图。Arrow 的 split、图像行号、图像名及第一条描述沿用原核验材料，85 份历史先验逐文件哈希与 E1 相符。调用实际 `BaseDataset.get_image` 时，同一图像的首、末描述样本均读取该图的同一份先验；记录见 `dataset_loader_replay.json`。170 条自身先验评分与上一轮的最大单项误差为 {j['scoring']['anchor_maximum_abs_error']:.9g}。

两份冻结模型的职责和输入来源如下，元信息由实际检查点读取：

| 模型 | SHA-256 | 训练配置中的缓存 |
|---|---|---|
| 下游 AGSE，本轮数量与检索打分使用 | `{plan['utility_model_sha256']}` | `/workspace/cup-debug/agent_cache_domain_tool` |
| E3-1 重建检索代理 | `{plan['proxy_sha256']}` | `/private/AGSE_E3_1_rebuild_20261112/plain_prior_full_20261112/run_01/priors` |

下游 AGSE 的历史缓存来源由保留的 `pipeline_domain_tool.sh`、原 rollout 源码和已核对的缓存文件共同追溯：历史流程为主 LoRA、场景工具、图像及原始首条描述。缓存按 split 和图像行号保存。该批历史缓存逐条生成时的实际停止 token 和完整轨迹为 `[待确认]`。

重建代理的纯推理缓存另有完整生成记录：基础 Qwen、工具集合为空、原始首条描述、贪心、每轮预算 64 token，取最后一次生成的末步隐藏状态。上一轮本次 EOS 对照使用 SFT-30、完整真实工具回填、统一中性输入和每轮 256 token。两套模型及缓存来源分别保留，当前数值对照固定采用下游 AGSE。

## 2. 数量组成解释汇总方向

以下为每图四条采样平均的 B PairAcc，所有失败保留：

{count_table}

确认 train 有 36/63 张双目标图和 5/63 张单目标图；既有 val 有 2/22 张双目标图和 13/22 张单目标图。两组内的单目标表现都下降、双目标表现都上升，原始总体变化因此朝相反方向。train 的数量 6/7 另含3张，完整结果保存在分组表中。

在两组共同数量 1、2、3、4 上等权汇总，train 60 图为 {mt['before_pairacc_B']:.2%} → {mt['after_pairacc_B']:.2%}，配对变化 95% 区间 [{100*mt['delta_pairacc_B_interval95'][0]:+.2f}, {100*mt['delta_pairacc_B_interval95'][1]:+.2f}] 个百分点；val 22 图为 {mv['before_pairacc_B']:.2%} → {mv['after_pairacc_B']:.2%}，区间 [{100*mv['delta_pairacc_B_interval95'][0]:+.2f}, {100*mv['delta_pairacc_B_interval95'][1]:+.2f}]。

共同数量范围的 train/val 改变量之差为 {100*d['split_delta_gap']:.2f} 个百分点。对称加权分解中的数量组成项为 {100*d['composition_component']:+.2f} 个百分点，数量组内差异项为 {100*d['within_count_component']:+.2f} 个百分点，两项之和还原总差异。这项分解描述当前数据的组成关系。

## 3. 共享状态与零输入对照

参考图按继承确认集顺序预先固定为第一张 `102.tif`，其标签停止与 EOS 贪心状态分别广播到全部85张目标图。每种状态在所有目标图中保持相同，目标图像本身沿用原输入。另用历史先验及零向量各评分85次。

{control_table}

自身 EOS 状态与共享 EOS 状态都使 train 准确率上升、val 准确率下降。共享状态仅提供固定参考图的信息，这个对照显示汇总增益包含普遍数量偏好的变化。零输入仍经过原 `agent_proj` 的线性层、偏置和归一化层，其定义及张量保存在归档中。

历史缓存为原始首条描述条件下的参考结果。原始描述可能直接给出数量词，相关记录保留在 `inputs/cases.json` 及 E1 核查材料中；本轮中性输入的 EOS 状态和共享状态分别承担新的对照职责。

![数量组成与共享状态对照](figures/eos_split_composition_and_shared_state.png)

## 4. 核验与结论

服务器与本机分别从保存特征复算 {v['float64_quantity_classifications_checked']} 次 float64 数量判别、{v['float64_val_ranks_checked']} 项排名、{v['stratified_rows_rebuilt']} 行分组结果、{v['bootstrap_intervals_rebuilt']} 组区间、4组数量等权结果和2组组成分解；上一轮425条 EOS 与935条 SFT 轨迹递归核验通过。统计按图像重采样10,000次；数量等权区间在各数量内重采样，主种子2026111451。单图或小样本分组同时保留图数与区间。

本阶段为观察到上一轮分歧后的探索性诊断。上一轮预设门槛及原始输出保持原值。当前证据支持将数量组成和普遍数量偏好列为主要处理项；后续收益核对采用数量均衡指标，并比较自身状态与固定共享状态。

## 5. 下一节点

已形成 [数量均衡短程试验方案](下一阶段数量均衡试验方案.json)：从当前确认 train 中按数量1、2、3、4各固定2图，共8图，拟执行32次实际优化更新，每数量8次、每图4次；其余55张train图用于训练图之外的确认，既有val22继续单列。训练图为 {', '.join(pilot)}。

奖励沿用 A 数量间隔及现有质量项，B用于评价。训练前先复核概率、完整工具序列与 EOS 末步状态一致性；检查点固定取最后一步。下一次实际运行前封存训练代码及完整数值门槛，确认数量宏平均、共享状态对照及协议质量。该方案当前为待执行设计。

复现入口见 [复现说明](复现说明_20261114.md)，核验见 [服务器核验](服务器独立核验.json)和 [本机核验](本机独立核验.json)。本轮新建证据包 `eos_split_diagnostic_evidence_20261114.tar.gz`，逐文件及本机备份核对见 [备份校验](备份校验.json)。
'''
(r/'EOS训练验证分歧诊断结论_20261114.md').write_text(text,encoding='utf-8')
(r/'下一节点.md').write_text('# 下一节点\n\n数量均衡的8图、32次实际更新短程试验；先封存代码并核对训练/推理一致性，再运行固定预算。采用数量宏平均与共享先验对照评价，55张训练图外确认与既有val22单列。具体方案见 `下一阶段数量均衡试验方案.json`。\n',encoding='utf-8')
print('DIAGNOSTIC_REPORT_AND_BALANCED_NEXT_DESIGN_WRITTEN')
