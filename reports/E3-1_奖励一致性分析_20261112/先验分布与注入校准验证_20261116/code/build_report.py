import argparse
import json
from pathlib import Path
from stage_io import read,dump
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;j=read(r/'analysis.json');full=read(r/'full_validation_metrics.json');proof=read(r/'本机独立核验.json');fp=read(r/'本机全量核验.json');assert proof['ok'] and fp['ok'];plan=read(r/'plan.json');align=read(r/'alignment_summary.json');co=j['groups']['confirmation55']['balanced32'];checks=j['adoption']['checks'];pre=read(r/'preflight.json')
def pp(x):return f"{x['mean']*100:+.2f}个百分点，95%区间[{x['interval95'][0]*100:+.2f}, {x['interval95'][1]*100:+.2f}]"
labels={'native':'原生纯推理先验','mean_shift_auto':'均值校准＋自身先验','learned_only':'仅可学习视觉提示','mean_shift_shared102':'均值校准＋固定共享先验'}
rows=['| 冻结前向方式 | I2T R@1 | R@5 | R@10 | T2I R@1 | R@5 | R@10 | mR |','|---|---:|---:|---:|---:|---:|---:|---:|']
for arm,label in labels.items():
 v=full['summary']['all210'][arm];rows.append('| '+label+' | '+' | '.join(f'{v[k]*100:.2f}' for k in ['I2T_R1','I2T_R5','I2T_R10','T2I_R1','T2I_R5','T2I_R10','mR'])+' |')
intervals='\n'.join('- 相对'+labels[x['reference']]+'：'+pp(x)+'。' for x in full['paired_mR_intervals'] if x['subset']=='unseen188')
qty=['| 前向方式 | 确认52图B全数量宏平均 | 原始55图等权 |','|---|---:|---:|']
for key,label in [('baseline_own','原生自身先验'),('selected_own','均值校准自身先验'),('selected_shared','均值校准共享先验'),('learned_only','仅可学习提示'),('original_cached','原始描述条件缓存')]:qty.append(f"| {label} | {co['groups'][key]['macro']['B_hard_correct']['mean']:.2%} | {co['groups'][key]['raw']['B_hard_correct']['mean']:.2%} |")
allv=full['summary']['all210'];newv=full['summary']['unseen188'];a=align['balanced32/confirmation55'];caption_count=fp['caption_queries'];rr=j['groups']['existing_val22']['balanced32']['groups'];newd=[x for x in full['paired_mR_intervals'] if x['subset']=='unseen188'];retrieval_pass=full['adoption']['passed']
nexttext='下一大阶段统一图像独立推理协议和训练端缓存：固定当前均衡第32步Agent及统一查询，复用本轮210张val缓存，为1680张train和210张test补齐1890份先验，形成2100图缓存；随后按相同划分、初始化与预算执行读出网络的匹配对照，完成val选择及保留test的最终检索评价。'
text=f'''# 先验分布、注入校准与完整检索验证（2026-11-16）

本阶段完成 **511份先验状态的分布与注入审计、5种冻结前向干预、55图数量确认、32次输入梯度探测，以及完整210图val检索验证**。核心审计包含1470条新评分；扩展验证新增188份贪心先验、2次锚点重放和840份图像编码。模型参数优化更新数为0。

完整val210的检索mR由 **{allv['native']['mR']:.2%} → {allv['mean_shift_auto']['mR']:.2%}**。此前未参与本轮评分的188图上，mR为 **{newv['native']['mR']:.2%} → {newv['mean_shift_auto']['mR']:.2%}**，对应配对变化 **{pp(next(x for x in newd if x['reference']=='native'))}**。数量确认集上，B全数量宏平均由 **17.65% → 23.16%**，变化 **{pp(co['contrasts']['selected_minus_native']['macro']['B_hard_correct'])}**。数量采用条件通过{sum(checks.values())}/5项；扩展检索条件通过{sum(full['adoption']['checks'].values())}/3项。

{nexttext}

## 1. 注入路径与分布证据

实际模型路径为：BF16隐藏态3584维 → 线性投影1024维 → LayerNorm → 重复成16个视觉提示 → 门控混合先验与可学习提示 → gather线性投影 → 插入CLIP视觉CLS之后 → 图像编码与adapter → L2归一化。源代码见 [AGSE前向](source_snapshot/vilt/modules/vilt_module.py)第584–605行、[CLIP插入](source_snapshot/prompt_clip/model.py)第240–275行。四份实际前向见证逐项重建了273个token的Transformer输入。

511份状态的原生门控截断值均为0，先验分支系数为1。32次原生/校准输入梯度探测均有非零响应，记录在 `gradient_witnesses.csv`；该探测围绕BF16状态的F32解量化点做epsilon0.1有限差分，策略更新数0。

原始描述缓存与均衡32步贪心先验的隐藏态平均范数分别为290.79和290.73，线性投影后平均范数分别为79.30和120.18。LayerNorm后的范数均约32；确认55图上，与同图原始缓存的投影方向余弦由 **{a['native_to_original_projected_cosine']:.4f}** 经均值校准提高至 **{a['aligned_to_original_projected_cosine']:.4f}**。方向比较来自已保存权重的float64重建，见 `alignment_by_image.csv`。

![分布与数量对照](figures/injection_domain_and_quantity.png)

## 2. 校准选择与数量确认

候选在评分前封存为原生、门控0、门控0.5、门控1、投影均值校准。均值校准偏置在各固定策略内，由原8训练图的“原始缓存线性投影均值−当前贪心先验线性投影均值”计算，并加在线性输出与LayerNorm之间。偏置拟合使用这8图，随后以均衡32步策略的A全数量宏平均和A最难负例间隔选候选；B未参与选择。程序在960条诊断网格后锁定 `mean_shift_auto`，再执行510条确认评分。

确认55图中，数量1–4共52图，各数量等权；其余数量保留图像等权结果。完整表如下：

{chr(10).join(qty)}

校准自身相对校准共享先验的B全数量变化为 **{pp(co['contrasts']['selected_own_minus_shared']['macro']['B_hard_correct'])}**，相对仅可学习提示为 **{pp(co['contrasts']['selected_minus_learned']['macro']['B_hard_correct'])}**。B最难负例间隔改善区间下界为正，二值判对与自身先验对照的采用条件尚未达成。SFT第30步支路作为补充保留，其全部表格与区间在 `analysis.json`。

## 3. 完整210图检索验证

在校准候选与偏置已经锁定后，原22张补充val图的RR由{rr['baseline_own']['retrieval_rr']['mean']:.6f}提高到{rr['selected_own']['retrieval_rr']['mean']:.6f}，据此增加完整val验证，并将新增188图单列。扩展阶段仍使用固定均衡32步策略、统一查询 `Describe the visible scene.`、真实场景工具和原生EOS控制器；复用22份已有val状态，新增188份贪心状态。两份val锚点重放与原状态精确一致。四路前向只改变已封存的校准/提示处理。

完整图库为210幅图和{caption_count}条原始描述。R@1/5/10与mR按仓库的重复token描述等价正例规则计算，实际调用原 `tr_top_k/ir_top_k` 完成一致性检查；token长度61，原有16个文本提示保留。保存的float32相似度矩阵与top-k索引为主结果，float64点积用于独立数值复核。

{chr(10).join(rows)}

mR为图像到文本、文本到图像两方向R@1、R@5、R@10的六项均值。对新增188图，以图像为配对重采样单位，文本查询随其所属图像成组抽样，保留原描述数权重。10,000次重采样的mR变化为：

{intervals}

![完整检索验证](figures/full_val210_retrieval.png)

val210是现有划分的验证集。此处新增188图指此前未参与本轮评分的图像；原test210保留供后续最终评价。本轮参数选择使用训练8图的A分数，扩展验证结果完整保留。

## 4. 缓存协议核对

85份原始domain-tool缓存的模型版本元数据为 `[待确认]`，文件SHA-256与历史标注逐项匹配。另核对现有 `plain_prior_full_20261112` 生成器：它对1680张train和210张val调用第一条标注描述 `item['caption']`，日志的LoRA路径为空、单轮token预算64；代码第185–217行和原始输入清单已保存于 `context_snapshot/`。本轮中性查询、均衡32步LoRA及原生EOS先验使用单独的来源记录。

因此，后续需要统一训练与部署的查询、策略版本、停止位置和缓存来源。校准对照展示了现有读出网络对先验分布变化的敏感性，数量信息及图像相关性继续由固定对照检验。

## 5. 双端核验与保存

服务器和本机分别重建511份原状态、线性投影、LayerNorm、门控及视觉提示，检查4份实际视觉插入、1470条新评分、340条父评分、220个原子集检索排名、672组区间与32次梯度见证。完整val阶段另核对210份先验、840份图像特征、{caption_count}个描述查询与9组图像成组区间。原AGSE全部张量哈希保持 `{read(r/'gpu_result.json')['original_model_parameter_hash']}`。

所有原始文件与失败轨迹保留。首轮在模型指纹的标量张量处理处停止，评分数0；v2只将张量展平后记录字节，新版随后完成全部评分。原失败日志、修正源码和哈希清单分别封存。

完整证据：`prior_injection_calibration_evidence_20261116.tar.gz`。校准偏置与前向处理见 `校准候选.json`；它保留数量条件和扩展检索条件的各项结果。双端备份、逐文件校验和复现说明分别见 `备份校验.json`、`file_checksums.json`、[复现说明](复现说明_20261116.md)。

## 6. 下一大阶段

{nexttext} 训练与验证各路共用数据划分、原始初始化和固定预算，比较原生图像独立先验、校准自身先验、校准共享先验和仅可学习提示；评估报告同时保留检索指标、数量确认及自身/共享先验对照。具体入口与预算在执行前按实际训练代码封存，规划见 [下一大阶段](下一大阶段方案_20261116.json)。
'''
(r/'先验分布与注入校准验证结论_20261116.md').write_text(text)
pt=__import__('torch').load(r/'原始运行记录/path/tensors.pt',map_location='cpu',weights_only=True)
dump(r/'校准候选.json',{'variant':'mean_shift_auto','injection_point':'agent_proj.1 forward input, after agent_proj.0 linear','offset_fit_images':plan['training_images'],'primary_policy':'balanced32','offsets':{k:v[0].tolist() for k,v in pt['offsets'].items() if k!='original_cached'},'selection':j['selection'],'quantity_adoption':j['adoption'],'full_retrieval_adoption':full['adoption'],'model_updates':0,'application':'frozen-forward candidate; original model files retained'})
dump(r/'下一大阶段方案_20261116.json',{'objective':nexttext,'official_splits':{'train':1680,'val':210,'test':210},'agent_policy':'balanced32','lora_hash':plan['policies'][1]['lora_hash'],'query':'Describe the visible scene.','cached_state_contract':'actual native-EOS last valid generate hidden; actual scene_classify feedback','reuse_current_val_states':210,'new_train_and_test_states':1890,'total_image_only_cache':2100,'readout_arms':['native image-only prior','mean-shift calibrated image-only prior','calibrated fixed shared prior','learned visual prompt only'],'agent_updates_during_cache_generation':0,'training_setup':'audit actual training entry, keep matched initialization/data/budget, seal exact trainable parameter list and schedule before launch','selection_split':'val210','final_split':'test210','report_node':'full cache, matched readout training, final retrieval evaluation and prior-specificity controls','conditioning_provenance':'current8 reference cached priors are first-caption-conditioned; preserved as calibration provenance'})
(r/'下一节点.md').write_text('# 下一大阶段\n\n'+nexttext+'\n');print('INJECTION_FULL_STAGE_REPORT_WRITTEN')
