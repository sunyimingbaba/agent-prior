"""Build an experiment delivery record from verified raw artifacts; no manuscript edits."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
def read(n):return json.loads((r/n).read_text())
j=read('analysis.json');d=read('protocol_plan.json');g=read('generation_result_v2.json');ev=read('evaluation_result.json');v=read('本机评估核验.json');gv=read('本机生成核验.json');b=read('本机逐图绑定核验.json');audit=read('EXPERIMENT_AUDIT.json')
assert all(read(n)['ok'] for n in ['本机生成核验.json','本机评估核验.json','本机描述统计核验.json','本机逐图绑定核验.json','final_source_audit.json']) and audit['checks']['result_existence']['status'].lower()=='pass'
primary='repository_duplicate_token_positive';fams=d['state_families'];labels={'legacy_last':'港口示例／末状态','legacy_answer_prefill':'港口示例／答案预填充','legacy_answer_pool':'港口示例／正文均值','selected_last':'顺序约定／末状态','selected_answer_prefill':'顺序约定／答案预填充','selected_answer_pool':'顺序约定／正文均值'}
def pct(x):return f'{x*100:.2f}'
def ci(x):return f"[{x['interval95'][0]*100:+.2f}, {x['interval95'][1]*100:+.2f}]"
def contrast(arm,f,kind='correct-minus-permutation_mean',protocol=primary):return next(x for x in j['retrieval_contrasts'] if x['arm']==arm and x['protocol']==protocol and x['contrast']==f+'/'+kind)
selected=j['protocols']['selected_validation'];key=contrast('calibrated_own','selected_answer_pool')
if key['interval95'][0]>0:state_conclusion='顺序约定下的正文均值在校准自身读出中呈现逐图配对收益，后续按训练划分开展专门的匹配读出与奖励实验。'
elif key['interval95'][1]<0:state_conclusion='顺序约定下的正文均值在校准自身读出中，正确配对低于置换均值；后续优先核对生成语义与先验投影的对应关系。'
else:state_conclusion='顺序约定下的正文均值与置换均值之差的区间覆盖0。后续RFPO奖励对照建议暂保留原末状态，先核对随机采样协议及同图候选奖励区分度；候选状态的专门适配另列E4验证。'
lines=['# 协议提示修复与候选状态验证（2026-11-17完成）','',f"本阶段采用训练63图选择提示，随后完成完整val210验证、六类状态及270组冻结读出对照。选定无示例的明确回合顺序约定（contract_only），train严格完成且消费已请求工具反馈62/63，val为{selected['strict_completed_with_consumed_tool']}/210；两者自行生成result标签分别0和{selected['generated_result_tags']}。Agent和读出端新增优化更新均为0。",'',f"在校准自身读出中，所选提示的正文均值正确配对减20次置换均值为{key['delta_mR']*100:+.2f}个百分点，名义配对95%区间{ci(key)}。{state_conclusion}",'','## 协议与历史监督核对','',
'历史SFT执行目标为15个工具调用与15个答案，共30例；每例6个活动标签，每轮180、两轮360。556个答案正文重叠位置均被屏蔽，工具反馈处于user上下文。正文来自既有冻结策略生成的轨迹。24个全词表NLL见证的CPU重算最大绝对差2.665e−7。',
'历史RL核对覆盖128条轨迹、8张图、32次更新、254轮、7076个生成token；126个实际工具事件及反馈记录在user回合，125条轨迹通过历史严格可用检查。原生成与replay逐位标签、长度一致，logprob最大数值差0.23154，符合原0.25阈值。SFT调用目标含EOS共6token，RL在闭标签处停止共5token。',
'30条SFT示例与128条RL轨迹均含同一港口示例。以上记录定位了训练覆盖情况；去示例错误形成于哪段训练、初始继承适配器更早的监督来源，均为[待确认]。完整审阅见 `.aris/traces/experiment-audit/` 与 `supervision_mask_audit.json`。','',
'三种提示、通过阈值与固定优先级在生成前封存。contract_only明确第一回合输出scene_classify调用，等待外部结果，再输出答案；neutral_placeholder保留中性格式示例；rotated_actual3按图像名确定训练示例。轮换来源为训练图101/1826/1902.tif，与273图队列分离。','',
'| 训练候选提示 | 严格完成并消费反馈 | 自行生成result标签 | 95%通过门槛 |','|---|---:|---:|---|']
for n,x in j['prompt_selection']['train_only_outcomes'].items():lines.append(f"| {n} | {x['strict_completed_with_consumed_tool']}/63 | {x['model_generated_result']} | {'通过' if x['passed'] else '未通过'} |")
lines+=['','选定规则仅使用train63协议结果。原始锁记录生成时尚无本阶段验证指标；30步格式SFT备选分支未触发，balanced32适配器哈希持续为 `'+d['policy']['lora_hash']+'`。','',
'每个工具反馈来自此前真实scene_classify轨迹的逐图文本及置信度，只有模型生成规范调用后才按需回放。外部程序保留user角色与result标签，当前阶段对分类器采用已核验结果回放。所有图像采用同一查询 `Describe the visible scene.`，贪心生成最多3轮、每轮256token。','',
'## 状态定义、位置与覆盖','',
'末状态沿用最后一个实际解码步的末层末位置。答案预填充取最终严格答案轮第0步，即请求反馈后答案提示的最后一个输入token表征。正文均值取输入token完整落在答案正文内的实际解码状态；生成token j在步骤j+1进入模型，以原始ByteLevel字节映射定位，边界跨越token排除。均值在CPU float64计算后转BF16。',
'严格答案或反馈条件未满足时，候选状态回退同轨迹末状态，并保留全部案例及回退标记。三类状态共用实际生成；原始每步矩阵、IDs、原生EOS、正文跨度与位置均保存。','',
'| 状态／val210 | 定义可用数 | 回退数 | 唯一状态数 | 中心化RMS | 有效秩 |','|---|---:|---:|---:|---:|---:|']
for f in fams:
 s=j['state_diagnostics'][f]['val'];lines.append(f"| {labels[f]} | {s['available']} | {s['fallbacks']} | {s['unique_states']} | {s['centered_rms']:.5f} | {s['effective_rank']:.3f} |")
lines+=['',f"V2保存{g['unique_generations']}次主队列生成、两张热身锚点与1638行状态库。273个原基线的末状态及生成IDs精确重放；独立核验重建{gv['candidate_state_tensors_rebuilt']}个候选张量、{gv['actual_decode_steps_verified']}步状态和{gv['body_input_positions_verified']}个正文输入位置。预评分与独立审计发现的严格答案悬空工具标签数为{gv['dangling_tool_tags_in_claimed_strict_answers']}。",'',
'初始V1在14条已完成试运行后，为补齐锚点矩阵、结果标签独立计数、单一答案限制及失败备选记录，停止该自有任务并新建V2。V1源码、日志和轨迹完整保留，14条IDs及末状态与V2精确一致；当时训练更新为0。修订清单见 `protocol_execution_addendum_v2.json`。','',
'2026-11-17恢复时，本机首先完整校验暂停归档1527份文件。版本化恢复入口保留651条旧记录的完整前缀、原提示选择锁和锚点；两条原提示锚点与一条已完成验证轨迹的生成IDs、全部解码状态及三类候选向量精确重放，随后只新增余下21张验证图。包版本与seed0 CUDA前反向见证保持一致，连续性独立核验见 `本机恢复连续性核验_20261117.json`。','','## 完整验证集与冻结读出','',
'四个读出检查点沿用父阶段val选择。原生自身及校准自身各132组，校准共享与可学习视觉提示各3组，共270组。每类状态均比较正确配对、训练102.tif同类共享状态及20次保持状态集合的划分内无固定点置换。val沿用同一组210图置换；train数量诊断采用63图队列内部置换。','',
'42组legacy_last验证矩阵在状态、检查点和文本特征逐字节证明后复用，数量特征重新计算。总来源行79650，其中新GPU图像编码70830、复用8820。数量库存56条，每图每句式按其对象只比较3–7个候选。','',
'| 状态 | 原生主mR | 校准自身主mR | 校准自身严格mR | 校准自身同分平均严格mR |','|---|---:|---:|---:|---:|']
for f in fams:lines.append(f"| {labels[f]} | {pct(j['retrieval'][primary]['native'][f]['correct']['mR'])} | {pct(j['retrieval'][primary]['calibrated_own'][f]['correct']['mR'])} | {pct(j['retrieval']['strict_image_correspondence']['calibrated_own'][f]['correct']['mR'])} | {pct(j['retrieval']['strict_tie_averaged']['calibrated_own'][f]['correct']['mR'])} |")
lines+=['','| 状态 | 原生正确−置换（百分点） | 配对95%区间 | 校准自身正确−置换（百分点） | 配对95%区间 |','|---|---:|---:|---:|---:|']
for f in fams:
 x=contrast('native',f);y=contrast('calibrated_own',f);lines.append(f"| {labels[f]} | {x['delta_mR']*100:+.2f} | {ci(x)} | {y['delta_mR']*100:+.2f} | {ci(y)} |")
lines+=['','![逐图对应与置换对照](figures/state_correspondence.png)','',
'主协议沿用重复token描述的等价正例；严格对应报告保存top-10前缀和完全同分项均匀抽取的期望命中率两个约定。全部六类状态、三协议、正确减共享及状态间变化共102组检索区间，另有136组分层数量区间。每项采用10000次配对重采样：验证图像及同图5条描述同步抽取，数量按真实1–4层分别抽取后等权。图库、模型及20个置换固定，区间为名义逐项95%。相同逐图命中轮廓先逐项相减再均值，保持精确零差。','',
'## 数量与投影诊断','',
'| 正确配对状态 | 原生val22 B宏平均 | 校准val22 B宏平均 | 校准train60 B宏平均 | 校准val门控均值 |','|---|---:|---:|---:|---:|']
for f in fams:
 q=j['quantity'];s=j['bank_path_diagnostics']['calibrated_own'][f];lines.append(f"| {labels[f]} | {pct(q['native'][f]['correct']['val']['B_hard_correct_macro_1to4'])} | {pct(q['calibrated_own'][f]['correct']['val']['B_hard_correct_macro_1to4'])} | {pct(q['calibrated_own'][f]['correct']['train']['B_hard_correct_macro_1to4'])} | {s['val_gate_mean']:.5f} |")
lines+=['','数量1–4样本数：val为13/2/3/4，train为5/36/14/5。85图原始人核标签、A/B句式的成对与全候选正确率、原始分差均保存；train另3图也保留，1–4宏平均对应train60。检索与数量使用原标注，状态离散度、格式通过率与门控属于描述性诊断；生成场景描述的语义正确性为[待确认]。',
'不同状态与固定读出端之间还存在输入分布变化，绝对指标结合各自的正确/置换/共享对照解释。本阶段val继续沿用读出选择所用划分，单策略与单组固定模型结果用于开发诊断。',
'本轮每图采用贪心生成，逐图配对对照检验图像对应作用。后续RFPO还需单独检验同一图像的多个采样候选是否得到有区分度的奖励。满足严格完成并消费反馈定义的纯预填充向量，由正文生成前的输入前缀决定；代码在轨迹结束后依完成情况选择这一向量或末状态回退。selected验证的4条异常实际采用末态，因此该家族同时包含纯预填充与末态回退。固定图像及相同工具历史的纯预填充，对后续正文候选共享同一数值；其逐图效果与同图候选奖励区分能力分别记录，并在进入策略训练前以采样实测核查。','',
'![覆盖与数量诊断](figures/state_coverage_and_counts.png)','',
'## 核验、保存与下一节点','',
f"服务器与本机分别重建{v['retrieval_values_rebuilt']}项检索指标、{v['quantity_summary_values_rebuilt']}项数量汇总及全部区间。补充核验逐行绑定{b['state_bank_rows_bound_to_frozen_cohort']}行状态库到封存cohort/Arrow图序，并将210条所选提示、五项关键输入字节绑定到原始来源。所有原始源码、模型、三份Arrow及父阶段证据复核原SHA；本阶段新增test评分为0。",'',
'独立审阅由experiment-audit技能要求的审阅代理完成，标注same-family/provisional；确定性重算与范围限定分别记录在 `EXPERIMENT_AUDIT.json`。完整运行记录、原始状态矩阵、统计脚本及报告在本机与/private保存，归档、分卷、文件索引及双端SHA证明见 `备份校验.json`。','',state_conclusion,'',
'本轮归入E4先验表征诊断及工具流程核查，为后续E3奖励试验提供生成条件和状态定义依据。按当前证据决定是否保留候选状态或先开展匹配读出检查；下轮RFPO须额外核对随机采样条件下的协议稳定性，并保持新旧奖励组的提示、状态提取、初始化、训练数据和预算一致。生成描述的事实语义另作盲法核查。下一轮方案在训练前单独封存，本轮完成后停止自动扩展实验范围。','']
(r/'协议提示修复与候选状态验证结论_20261117.md').write_text('\n'.join(lines))
nextplan={'objective':'依据本轮E4补充诊断设计后续E3奖励对照','prompt':'contract_only','Agent_updates_this_stage':0,'readout_updates_this_stage':0,'evidence':{'selected_validation_protocol':selected,'calibrated_own_selected_pool_correct_minus_permutation':key},'decision':state_conclusion,'actions':['use fixed-model contrasts to assess candidate state evidence','check protocol stability and within-image reward variance under RFPO stochastic sampling','conditionally verify state/readout distribution compatibility','blind-check image/tool/answer semantics','seal same-initialization same-budget old/cross-modal/hard-negative reward comparison separately'],'development':['train','val'],'test_policy':'retain existing test artifacts; no new test scoring in this stage'}
(r/'后续RFPO实验依据与待验证项_20261117.json').write_text(json.dumps(nextplan,ensure_ascii=False,indent=2)+'\n');print('VERIFIED_PROTOCOL_STATE_REPORT_WRITTEN')
