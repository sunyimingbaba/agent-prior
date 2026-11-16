"""Generate a factual diagnostic delivery note from dual-verified results."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
def read(n):return json.loads((r/n).read_text())
j=read('analysis.json');d=read('diagnostic_plan.json');ep=read('evaluation_plan.json');g=read('generation_result.json');sv=read('server_verification.json');lv=read('本机独立核验.json');pv=read('本机状态路径核验.json')
assert sv['ok'] and lv['ok'] and pv['ok'] and read('final_source_audit.json')['ok']
assert read('EXPERIMENT_AUDIT.json')['checks']['result_existence']['status'].lower()=='pass'
base=ep['baseline'];primary='repository_duplicate_token_positive';names=[c['name'] for c in d['conditions']];labels={'native':'原生自身','calibrated_own':'校准自身','calibrated_shared':'校准共享','learned_visual_only':'可学习视觉提示'}
def pct(x):return f'{x*100:.2f}'
def cond(n):
 c=next(c for c in d['conditions'] if c['name']==n)
 return ('有示例' if c['example']=='with_example' else '去示例')+'／'+('真实图' if c['policy_image']=='real' else '灰图')+'／'+('逐图反馈' if c['feedback']=='recorded_real' else '常量反馈')
def rc(arm,protocol,name):return next(c for c in j['retrieval_contrasts'] if c['arm']==arm and c['protocol']==protocol and c['contrast']==name)
def bounds(c):return f"[{c['interval95'][0]*100:+.2f}, {c['interval95'][1]*100:+.2f}]"
own_perm=rc('calibrated_own',primary,'permutation_mean');delta=own_perm['delta_mR']*100
noexample_failures=sum(g['quality_by_condition'][c['name']].get('model_generated_tool_result',0) for c in d['conditions'] if c['example']=='without_example')
if own_perm['interval95'][1]<0:conclusion='校准自身模型的正确图像配对优于本轮20次置换均值，逐图对应信息在该固定读出端呈现正向证据。'
elif own_perm['interval95'][0]>0:conclusion='校准自身模型的置换先验均值高于正确配对，后续优先核查状态与读出端的对应关系。'
else:conclusion='校准自身模型的置换差异区间覆盖0，逐图先验对应信息的额外收益继续待验证。'
lines=['# 图像依赖性与提示对照诊断结论（2026-11-16）','',
f"本阶段完成273图三因素八条件、20次完整划分内先验置换及64组冻结读出前向。生成逻辑状态2184份：复用基线273份，实际新增生成{g['new_actual_generations']}次，同条件等价输入复用{g['canonical_reused']}份，另精确重放两张锚点。Agent与读出端新增优化更新均为0。",'',
f"校准自身模型在完整val210上的主协议基线mR为{pct(j['retrieval'][primary]['calibrated_own'][base]['mR'])}%；20次置换的均值相对基线变化{delta:+.2f}个百分点（相对变化{own_perm['delta_mR']/j['retrieval'][primary]['calibrated_own'][base]['mR']*100:+.2f}%），配对95%区间{bounds(own_perm)}。{conclusion}",'',
f'去示例四条件累计{noexample_failures}个逻辑案例被控制器标记为模型自行输出工具结果标签；这些条件的实际工具请求为0。有示例四条件中，每个逻辑案例均保留一次模型工具请求及已回放反馈；灰图加常量反馈按等价输入复用相同计算。提示示例的协议作用成为本轮新增的直接证据，代表轨迹见 `protocol_witnesses.json`。','',
'## 1. 实验对象、输入控制与范围','',
'完整val210用于检索及状态诊断，train63沿用原数量清单。数量评分覆盖原85图，其中val22用于验证子集分析；各对象及句式只比较对应的3–7个数量候选，共用56条固定文本库存。人工标签原表副本及SHA见 `inputs/label_provenance/`。','',
'三因素分别为固定港口示例有/无、Agent图像真实/灰色127、工具反馈逐图记录/训练102.tif常量。灰图保留256×256尺寸，检索骨干持续读取真实图像。工具反馈只在模型请求scene_classify时回放，逐图文本及置信度取自已核验的原始真实分类轨迹；常量取训练102.tif。最终轮请求后尚未进入下一次生成的反馈另计，反馈暴露情况见第4节。','',
'20次置换在完整train1680与val210中分别生成，拒绝所有含固定点的置换；每次保持先验向量多重集合完全相同。训练数量子集按完整训练划分的置换取源向量。四个读出检查点沿用上一阶段val选择，全部参数冻结，原生与校准自身各28条件，共享与可学习视觉提示各4个实现对照。','',
'父阶段完整性核验读取既有test产物和三份Arrow的字节哈希；本阶段的生成、评分与方案比较使用train/val。旧test指标及评分矩阵保持封存。当前val沿用读出模型选择所用划分，结果用于固定模型诊断。','',
'## 2. 完整val210检索结果','',
'主协议沿用重复token描述的等价正例。原严格图像对应使用保存的top-10前缀；补充协议对每个K处的完全同分项按等概率选择计算期望命中率。并列处理方案在新评分结果产生前单独封存，见 `tie_sensitivity_plan.json`。','',
'| 生成条件 | 原生自身主mR | 校准自身主mR | 校准自身严格mR | 校准自身同分平均严格mR |','|---|---:|---:|---:|---:|']
for n in names:lines.append(f"| {cond(n)} | {pct(j['retrieval'][primary]['native'][n]['mR'])} | {pct(j['retrieval'][primary]['calibrated_own'][n]['mR'])} | {pct(j['retrieval']['strict_image_correspondence']['calibrated_own'][n]['mR'])} | {pct(j['retrieval']['strict_tie_averaged']['calibrated_own'][n]['mR'])} |")
lines+=['','| 固定读出 | 协议 | 20次置换mR均值 | 相对正确配对变化（百分点） | 配对95%区间 |','|---|---|---:|---:|---:|']
for arm in d['primary_readouts']:
 for protocol,label in [(primary,'仓库主协议'),('strict_image_correspondence','原严格'),('strict_tie_averaged','同分平均严格')]:
  c=rc(arm,protocol,'permutation_mean');s=j['permutation_summary'][arm][protocol];lines.append(f"| {labels[arm]} | {label} | {pct(s['mean_mR'])} | {c['delta_mR']*100:+.2f} | {bounds(c)} |")
native_strict=rc('native','strict_image_correspondence','permutation_mean');native_tie=rc('native','strict_tie_averaged','permutation_mean')
lines+=['',f"原生读出中，原严格对应的置换变化为{native_strict['delta_mR']*100:+.2f}个百分点、区间{bounds(native_strict)}；同分平均后为{native_tie['delta_mR']*100:+.2f}个百分点、区间{bounds(native_tie)}。严格对应的区间判断随并列处理约定发生变化，后续同时核对两者。"]
lines+=['','置换均值的配对区间按验证图像成组重采样10000次，同图5条描述共同抽样，图库固定；20次置换之间的标准差、极值及逐次指标单列在 `analysis.json`。区间对应固定模型及这20次置换的图像抽样不确定性。所有对照与名义逐项95%区间均保留。','',
'![冻结先验对照与配对区间](figures/frozen_prior_contrasts.png)','',
'## 3. 三因素平均变化与数量诊断','',
'去示例条件未请求工具，分配的两类反馈均未进入实际生成；反馈因素的平均变化需结合这一暴露情况解释。下表中的每个因素变化，均对另外两个因素的四种组合取平均。简单条件相对基线的区间及全部186组检索区间、248组数量区间见 `analysis.json`。','',
'| 校准自身读出因素变化 | 主协议mR变化（百分点） | 配对95%区间 |','|---|---:|---:|']
for field,label in [('example','去除固定示例'),('policy_image','Agent图像改为灰图'),('feedback','改为训练102.tif常量反馈')]:
 c=rc('calibrated_own',primary,'factor/'+field);lines.append(f"| {label} | {c['delta_mR']*100:+.2f} | {bounds(c)} |")
lines+=['','| 生成条件 | 原生val22 B宏平均 | 校准val22 B宏平均 | 校准train60 B宏平均 |','|---|---:|---:|---:|']
for n in names:lines.append(f"| {cond(n)} | {pct(j['quantity']['native'][n]['val']['B_hard_correct_counts1to4_macro'])} | {pct(j['quantity']['calibrated_own'][n]['val']['B_hard_correct_counts1to4_macro'])} | {pct(j['quantity']['calibrated_own'][n]['train']['B_hard_correct_counts1to4_macro'])} |")
lines+=['',f"校准自身模型的八个生成条件在val22的B全候选宏平均均为{pct(j['quantity']['calibrated_own'][base]['val']['B_hard_correct_counts1to4_macro'])}%。",'',
'数量1–4等权，val样本数为13/2/3/4，train为5/36/14/5。train60属于读出训练内诊断；其余3张训练图的全部指标也保留。数量配对区间在每个真实数量层内同步重采样，A/B句式、成对与全部候选指标及原始分差完整保存。','',
'## 4. 原始状态、轨迹与读出路径','',
'| val生成条件 | 唯一状态数 | 中心化RMS | 轨迹可用规则通过 | 实际请求工具图数 | 已进入后续生成的反馈数 |','|---|---:|---:|---:|---:|---:|']
for n in names:
 s=j['generation_states'][n]['val'];lines.append(f"| {cond(n)} | {s['unique_state_sha256']} | {s['centered_rms']:.4f} | {s['quality_usable']}/210 | {s['tool_requested_images']}/210 | {s['feedback_consumed_events']} |")
lines+=['','| 固定读出与条件 | 门控均值 | 门控等于1比例 | 图像特征最大绝对变化 |','|---|---:|---:|---:|']
for arm in d['primary_readouts']:
 for n in [base,names[2],names[4],names[7]]:
  v=j['readout_path'][arm][n];lines.append(f"| {labels[arm]}：{cond(n)} | {v['gate_mean']:.4f} | {v['gate_one_fraction']*100:.2f}% | {v['image_feature_max_abs_change']:.3g} |")
lines+=['','轨迹可用规则按完成答案、格式及预算检查计算，异常轨迹与状态全部保留。词项出现率属于生成文本描述统计。原始向量的离散程度、投影变化、门控饱和率及图像特征变化分别保留，检索与数量任务按真实标签评分；轨迹统计记录协议执行和词项出现，生成描述的语义正确性标记为[待确认]。','',
'原代码的提示融合为 `(1−g)×Agent先验提示 + g×可学习视觉提示`。两种自身读出在本轮val条件中g均为0，当前路径使用Agent投影提示；后续继续核查该状态及投影保留的图像信息。','',
'四个基线的图像/文本特征与父阶段精确一致。共享及可学习视觉提示两个实现对照各4条件的图像特征完全一致。独立核验用原检查点的线性层、LayerNorm和门控权重从已保存输入重新计算路径，同时核对全部18880个先验来源及数值SHA。','',
'![状态路径与数量诊断](figures/state_path_and_count_diagnostics.png)','',
'## 5. 核验与保存','',
f"服务器与本机分别核验2184条生成记录及64组冻结前向的保存结果，重建40个完整划分置换，并从原始张量重算{lv['retrieval_metric_values_rebuilt']}项检索指标、{lv['quantity_summary_values_rebuilt']}项数量汇总、186组检索区间及248组分层数量区间。float64重算缩放分数的最大绝对误差为{lv['float64_score_max_abs_error']:.3g}，投影为{lv['float64_projection_max_abs_error']:.3g}，门控为{lv['float64_gate_max_abs_error']:.3g}。原始源码、模型、数据及父阶段证据保持原哈希。",'',
'独立审阅及原始响应见 `EXPERIMENT_AUDIT.json` 与 `.aris/traces/experiment-audit/`；语义审阅标注same-family/provisional。单种子、既定读出检查点、验证图库及小规模数量子集构成本轮证据范围。完整原始记录在本机与/private双端保存，SHA见旁路 `备份校验.json`；既有四组读出检查点归档继续由父阶段备份引用。','',
'## 6. 下一阶段安排','',conclusion,'',
'下一轮优先核查工具结果角色与训练监督掩码，比较中性格式示例、多场景轮换示例与当前提示，验证实际工具请求、真实回填及完成答案的比例。随后固定模型比较候选状态的图像信息可读性：以当前末有效解码状态作基线，核对实际生成中工具回填后首步及答案片段状态的位置和数值，保持相同图像、查询及预算。通过正确配对、同分布置换、共享状态三组对照检查检索与数量变化，再据证据确定状态与奖励方案。候选状态的位置定义和精确预算在控制器核对后封存，train/val用于开发，既有test记录继续保存。','']
(r/'图像依赖性与提示对照诊断结论_20261116.md').write_text('\n'.join(lines))
nextplan={'objective':'优先修复对固定示例的工具协议依赖，再检验候选状态与读出对应信息，形成奖励方案','trigger':{'calibrated_own_permutation_minus_baseline_pp':delta,'interval95_pp':[x*100 for x in own_perm['interval95']],'conclusion':conclusion},'protocol_evidence':{'no_example_model_generated_result_logical_cases':noexample_failures,'actual_tool_requests_without_example':0},'protocol_actions':['audit assistant/tool-result supervision masks','neutral tool-format example','scene-rotated examples','verify requested actual tool feedback and completed answers before state/reward selection'],'fixed_assets':'same balanced32 Agent and all frozen parent-selected readouts','candidate_states':['current last actual valid generate-step state','first answer-generation step after recorded tool feedback','answer-content state aggregation after exact input-token position audit'],'controls':['correct image-state correspondence','within-split distribution-preserving permutation','constant/shared state'],'development_splits':['train','val'],'budget':'seal after auditing candidate state positions; reuse the same generation replay for multiple state extractions where possible','new_Agent_or_readout_updates_authorized_in_this_diagnostic_stage':0,'test_record':'existing results remain archived','report_node':'state image-information/readout correspondence and concrete reward choice with paired evidence'}
(r/'下一阶段状态与奖励方案_20261116.json').write_text(json.dumps(nextplan,ensure_ascii=False,indent=2)+'\n');print('VERIFIED_DIAGNOSTIC_REPORT_WRITTEN')
