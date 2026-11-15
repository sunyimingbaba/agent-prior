import argparse
from pathlib import Path
from stage_io import read,dump
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;j=read(r/'analysis.json');plan=read(r/'plan.json');proof=read(r/'本机独立核验.json');assert proof['ok'];names=[p['name'] for p in plan['policies']];labels=['SFT第30步','数量均衡第32步']
def pp(x):return f"{x['mean']*100:+.2f}个百分点，95%区间[{x['interval95'][0]*100:+.2f}, {x['interval95'][1]*100:+.2f}]"
def fp(x):return f"{x['mean']:+.8f}，95%区间[{x['interval95'][0]:+.8f}, {x['interval95'][1]:+.8f}]"
table=['| 固定策略 | 自身先验B全数量判对率 | 其余7图先验均值 | 自身−其余7图 |','|---|---:|---:|---:|'];cover=['| 固定策略 | 历史4条候选上限 | 新采样前4条 | 新采样16条 | 前4→16变化 |','|---|---:|---:|---:|---:|'];quality=['| 固定策略 | 质量合格率 | 严格格式合格率 | 实际工具率 | 16候选B判别变化图数 |','|---|---:|---:|---:|---:|'];same=['| 固定策略 | 6张飞机图：自身−同对象5图 | B最难负例间隔变化 |','|---|---:|---:|']
for name,label in zip(names,labels):
    x=j['policies'][name];z=x['native_vs_swapped']['new_sample']['B_hard_correct'];table.append(f"| {label} | {z['own']['mean']:.2%} | {z['donor']['mean']:.2%} | {pp(z['own_minus_donor'])} |")
    c=x['coverage'];cover.append(f"| {label} | {c['historical4']['qualified_B_oracle_success']['mean']:.2%} | {c['new4']['qualified_B_oracle_success']['mean']:.2%} | {c['new16']['qualified_B_oracle_success']['mean']:.2%} | {pp(x['coverage_expansion']['qualified_B_oracle_success'])} |")
    q=c['new16'];quality.append(f"| {label} | {q['quality']['mean']:.2%} | {q['strict_quality']['mean']:.2%} | {q['tool_usage']['mean']:.2%} | {q['B_decision_varies_images']}/8 |")
    same.append(f"| {label} | {pp(x['same_object_airplanes']['new_sample']['B_hard_correct'])} | {fp(x['same_object_airplanes']['new_sample']['B_hard_margin'])} |")
quality_ok=all(j['policies'][p]['coverage']['new16']['quality']['mean']>=.90 for p in names)
signal=[p for p in names if j['policies'][p]['native_vs_swapped']['new_sample']['B_hard_correct']['own_minus_donor']['interval95'][0]>0 and j['policies'][p]['native_vs_swapped']['new_sample']['B_hard_margin']['own_minus_donor']['interval95'][0]>0]
if not quality_ok:route='protocol_failure_audit';nexttext='下一阶段固定本轮质量失败轨迹，检查停止原因、工具回填和数量候选覆盖的关系，保持0次策略更新。'
elif signal:route='matched_reward_short_pilot';nexttext='下一阶段针对已通过自身/置换数量信号核对的固定策略，封存最难负例与原句对奖励的匹配短训对照；先固定预算、种子和模型选择规则，再启动。'
else:route='original_cache_and_prior_injection_audit';nexttext='下一阶段核对原始描述条件缓存与当前纯推理先验的分布及冻结AGSE注入路径：复用已核验的8份原始缓存，补做8个来源先验×8幅目标图的64条评分，并与本轮保存的2176条评分对照；保持0次策略更新。'
cache=read(r/'original_cache_inventory.json');pre=read(r/'preflight.json');capture=read(r/'原始运行记录/runtime_evidence/utility_python_sources_after.json');qdiff=len(proof['float64_multi_count_decision_changes']);native_delta=j['policy_comparison_native']['B_hard_correct'];margin_lines='\n'.join(f"- {label}：自身−其余7图的B最难负例间隔变化为 {fp(j['policies'][name]['native_vs_swapped']['new_sample']['B_hard_margin']['own_minus_donor'])}。" for name,label in zip(names,labels))
text=f'''# 采样覆盖与跨图先验置换结论（2026-11-15）

本阶段完成 **256条新评估采样、2次贪心锚点探测及2176条来源图—目标图评分**，策略优化更新数为 **0**。固定SFT第30步与均衡第32步两个检查点，各使用原8训练图、每图16个固定种子；另复用16条已保存贪心轨迹。服务器和本机独立重建评分、完整置换网格、选样与区间。

两策略将新候选从前4条扩到16条后，质量合格候选的B全数量判对覆盖都保持25.00%，配对变化为0。自身先验相对其余7图先验的数量判对变化分别为 **{pp(j['policies'][names[0]]['native_vs_swapped']['new_sample']['B_hard_correct']['own_minus_donor'])}** 和 **{pp(j['policies'][names[1]]['native_vs_swapped']['new_sample']['B_hard_correct']['own_minus_donor'])}**。

{nexttext}

## 1. 问题、样本和冻结条件

上一轮离线分析显示，固定均衡第32步策略在原8训练图的每图4个候选间，B全数量二值判别变化为0/8。本轮将每图候选扩到16条，并对同一目标图逐一替换8个来源图的先验，直接核对数量评分对先验来源的敏感性。图像固定为 {', '.join(plan['images'])}，数量1–4各2图，其中6张飞机图、1张沙坑图和1张储罐图。

检查点哈希分别为 `{plan['policies'][0]['lora_hash']}` 和 `{plan['policies'][1]['lora_hash']}`。各策略在载入后与生成结束均核对LoRA哈希，所有参数冻结。实际硬件为 {pre['gpu']}、81920MiB，驱动{pre['driver']}，PyTorch {pre['packages']['torch']}、Transformers {pre['packages']['transformers']}，沿用原 `/private` GPFS与A100环境。方案SHA-256为 `{j['plan_sha256']}`。

生成查询为 `Describe the visible scene.`，沿用既有系统示例、真实 `scene_classify` 回填和回答原生EOS终止。采样温度1、top_k=0、top_p=1、min_new_tokens=1，最多3轮、每轮256 token。种子为 `2100000 + 策略序号×10000 + 图像序号×16 + 采样序号`。保存实际generate最后有效解码步、末层末位置的BF16[3584]，同一状态直接传入冻结下游AGSE。失败轨迹完整保留。

两次贪心锚点分别重放102.tif；两策略的token序列和数值状态哈希均与已保存贪心结果逐项一致。前置探测单列2次，正式新评估采样为256条。原始80条历史轨迹还包含每图每策略4条历史采样，用于单列覆盖比较。

## 2. 完整置换网格与指标

对于每个固定策略及采样序号，将8幅来源图各自产生的先验分别注入8幅目标图，形成8×8网格，共2×16×8×8=2048条评分；16份贪心来源状态再形成128条评分，总计2176条。评分时目标图像和目标数量候选固定，先验保留来源图、状态哈希、工具和质量来源。

沿用原56条冻结数量文本特征。B全数量判对要求人工数量对应的B句分数严格超过同对象所有已缓存错误数量；B最难负例间隔为该正例分数减去最高错误数量分数。每幅目标图先平均16个采样序号，再对数量1–4等权汇总；自身先验与其余7个来源先验的均值在目标图内配对比较。

{chr(10).join(table)}

{margin_lines}

![采样覆盖与置换结果](figures/coverage_and_prior_permutation.png)

完整来源—目标均值矩阵如下，橙色对角线表示来源图与目标图相同：

![来源与目标图完整矩阵](figures/donor_target_B_accuracy.png)

## 3. 同对象对照与固定策略差异

全网格包含对象类别不同的来源先验，因此另在6张飞机图中，以自身先验对照其余5张飞机来源图的平均先验。该对照按6幅图像等权汇总：

{chr(10).join(same)}

两个固定策略的新采样自身先验B全数量宏平均变化（均衡第32步−SFT第30步）为 **{pp(native_delta)}**。两策略使用各自预先固定的种子段，策略比较按相同目标图配对，各自生成轨迹完整保存。

## 4. 4条扩到16条的候选覆盖

候选上限定义为每幅图是否至少存在一条“质量合格且B全数量判对”的候选。新前4条是同一组16条的固定前缀；历史4条使用上一阶段封存的种子，单列呈现。两策略16条候选中的合格B判对均只覆盖121.tif与124.tif两幅数量为2的图，其余6图在本次采样内的合格B判对覆盖为0：

{chr(10).join(cover)}

{chr(10).join(quality)}

A句对与A最难负例奖励仍按 `0.2×质量合格+A间隔` 选样。B预言机使用B最难负例间隔描述候选池上限；全部240行选样和逐图覆盖见CSV。每个图像的连续间隔极差与A/B组内排序一致性另存于 `coverage_by_image.csv`。生成文本的语义和计数事实准确性在 [256条事实清单](256条文本事实复核清单.csv) 中保持 `[待确认]`。

## 5. 区间与独立核验

统计按图像重采样10,000次，原8图数量宏平均在各数量内抽样，每数量2图；飞机子集以6图等权抽样。主种子2026111557，具体label派生种子、逐图差值和全部区间保存在 `paired_statistics.csv`。区间描述当前固定图像及候选集合的变化。

独立程序核对256条新评估轨迹、80条历史轨迹、2次锚点探测、2176条来源网格、4352个A/B数量判别、{proof['bootstrap_intervals_rebuilt']}组区间及240行选样。保存图像/文本特征的float64点积与冻结float32评分最大差为{proof['float64_quantity_dot_max_error']:.3g}，全数量判别及预测数量的数值复核变化为{qdiff}项。16条贪心对角线评分与父阶段的最大误差见 `scoring/result.json`。

原32份检查点文件、基座权重、原代码、运行时兼容文件与父阶段输入经过结束核对。冻结效用模型相关原Python源码共{capture['file_count']}份，在评分前后逐项哈希一致；相关清单保存在 `runtime_evidence/utility_python_sources_*.json`。

## 6. 下一节点与保存

{nexttext}

原始缓存文件的8个路径和SHA-256已核对，清单见 `original_cache_inventory.json`。后续条件和路线按数值评分前固定的 `decision_rules.json` 执行；本轮路线为 `{route}`。完整下一阶段设计见 [下一阶段预案](下一阶段预案_20261115.json)。

本阶段完整证据为 `quantity_sampling_permutation_evidence_20261115.tar.gz`，服务器目录 `{plan['remote_root']}`；逐文件校验和双端备份见 `file_checksums.json` 与 `备份校验.json`。复现步骤见 [复现说明](复现说明_20261115.md)。前序两个模型和优化器继续由既有检查点归档保存。
'''
(r/'采样覆盖与先验置换结论_20261115.md').write_text(text);(r/'下一节点.md').write_text('# 下一节点\n\n'+nexttext+'\n')
nextplan={'status':'frozen_after_current_diagnostic','route':route,'reason':{'quality_gate_passed':quality_ok,'policies_passing_native_quantity_signal':signal,'B_native_minus_donor':{p:j['policies'][p]['native_vs_swapped']['new_sample']['B_hard_correct']['own_minus_donor'] for p in names}},'parent_stage':plan['remote_root'],'parent_plan_sha256':j['plan_sha256'],'original_cache_root':cache['root'],'original_cache_files':cache['records'],'frozen_utility_sha256':plan['utility_model_sha256'],'prior_source_python_inventory':'runtime_evidence/utility_python_sources_after.json','proposed_next_work':nexttext,'human_text_fact_accuracy':'[待确认]'}
if route=='original_cache_and_prior_injection_audit':nextplan.update(optimizer_updates=0,new_Qwen_generations=0,new_original_cache_full_grid_scores=64,reused_current_grid_scores=2176,analyses=['compare original caption-conditioned cached priors with current inference priors: hidden and projected prompt norms/cosines','trace actual frozen AGSE prior projection and injection path with source lines and fixed input witnesses','same target image original versus new own/foreign states; preserve all8x8 original-cache donor scores','report text-conditioning and model/source provenance with original first caption logged; compare quantity ranks on same frozen56 text features'])
if route=='original_cache_and_prior_injection_audit':
    nextplan.update(planned_remote_root='/private/AGSE_E3_1_rebuild_20261112/prior_injection_audit_20261115',cached_prior_policy_provenance='[待确认：按原始缓存归档记录核对]',bootstrap_replicates=10000,bootstrap_seed=2026111558,interpretation='Report original first-caption conditioning, policy provenance and paired score changes explicitly. Identify projection/injection tensors from the actual runtime module tree and saved source files.')
dump(r/'下一阶段预案_20261115.json',nextplan);print('PERMUTATION_REPORT_WRITTEN',route)
