from pathlib import Path
import csv
import json

root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((root/'baseline_summary.csv').open(encoding='utf-8-sig')))
lookup={(s['model'],s['split'],s['task']):s for s in rows}
names={'CLIP':'原始 CLIP','CUP_CCB':'CUP（CCB，项目复现）','ASPF':'ASPF（纯推理重建）','AGSE':'原始 AGSE 权重，通用查询','AGSE_archived':'原始 AGSE 权重，归档首条描述先验','CLIP_legacy':'旧 CLIP，16 个零视觉提示'}
titles={'val':'验证图','train':'训练图','all':'全部诊断图'}
lines=['# UCM 上的数量、属性与空间关系诊断','',
'本轮在 A100 上冻结各方法，对同一批图像及同一批正负描述逐项打分。CUP 的直接对照是归档的 CCB 聚类先验版；原始 CLIP 另列作骨干参考。[基线来源](基线来源.md)','',
'**当前最清楚的发现是原始 AGSE 对查询输入敏感。** 每张图输入相同的通用查询时，133/133 张图的首个原生生成词元都是结束符，0 张实际调用场景工具；读出层仍可得到图像条件下的隐藏状态。6 张预先指定的图改用各自首条原始描述后，全部产生可见文本、5 张调用工具，6 个最终隐藏向量与历史归档逐元素相等。6 条最终解码文本都含有异常串 addCriterion，工具调用的恢复仍伴随回答异常。这个对照固定了权重、图像和设备。首条描述中可能给出目标细节，历史输入与通用查询的差异也包含措辞、生成和工具路径变化。[原生词元与缓存对照](native_token_control.json)','',
'每格为“正确数/图数（平均余弦分差）”。分差是正确描述得分减错误描述得分，大于 0 计正确。数量 A、B 是同图两套措辞。属性、关系分别按图像报告。各方法使用余弦相似度，跨模型的原始分差也受各自嵌入尺度分布影响。']
for sp,label in titles.items():
    lines.extend(['',f'## {label}','','| 方法 | 数量 A | 数量 B | 属性 | 空间关系 |','|---|---:|---:|---:|---:|'])
    for method,display in names.items():
        cells=[]
        for task in ('quantity_A','quantity_B','attribute','relation'):
            v=lookup[method,sp,task]
            cells.append(f"{v['correct']}/{v['images']}（{float(v['mean_cosine_margin']):+.4f}）")
        lines.append('| '+' | '.join([display,*cells])+' |')
lines.extend(['','## 样本和解释','',
'数量沿用用户确认的 100 张 UCM 图，含 78 张训练图、22 张验证图。64 张的首条原始描述含有正确数量词。属性为 22 个图文判断（12 train、10 val），空间关系为 20 个（10 train、10 val）。新增样本由助手在评分前看图选定，同场景两图共用两条互为正负的描述，每种措辞在一张图上为正、另一张为负。新增样本的独立人工确认待补。数量正负数词频率不均，词频偏好可能影响得分。数量复用图与新样本有交集，共涉及 133 张不同图（train 96、val 37）。未读取 test 作本轮评分。UCM 原划分也可能含同一地点的相邻裁块。',
'',
'验证集上，同场景反向配对的空间关系共 5 对图。CUP 和通用查询 AGSE 各对 5/10，且两者各自都是每对只判对一张。这样的结果与对一对图始终偏好同一句描述兼容。属性验证只有 5 对，CUP 对 3/10，通用查询 AGSE 对 6/10。现阶段宜把空间关系作为扩大独立复核的候选，把 AGSE 的通用查询终止问题作为单独的生成诊断。现有组间差距包含读出器与训练历史等差异。',
'','## 复算与标注材料','',
'各组输出的归一化图像和文本特征保存在 outputs/。本地独立以双精度点积逐项重算 1,452 项判断。两套历史对照的各 200 条数量正误逐项吻合。归档和通用查询分组、读出器状态、工具调用、输入图像以及数据集划分都保留在原始文件中。',
'','[逐图得分](pair_scores.csv) · [分组统计](baseline_summary.csv) · [反向配对](reversal_check.csv) · [分数复核](score_verification.json) · [权重与输入哈希](asset_bindings.json) · [逐图复核材料](标注核验.md)'])
(root/'实验结果.md').write_text('\n'.join(lines)+'\n')
