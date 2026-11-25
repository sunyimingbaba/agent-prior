"""Attach the final controlled measurements and actual training eligibility to the Chinese report."""
from pathlib import Path
from rosters import ROOT,read
def exists(k):return (ROOT/'results'/(k+'.json')).is_file()
def get(k):return read(ROOT/'results'/(k+'.json'))
def diff(v):return f"{100*v['mean']:+.2f} [{100*v['interval'][0]:+.2f}, {100*v['interval'][1]:+.2f}]"
def rate(v):return f"{100*v['mean']:.2f}%"
def main():
    path=ROOT/'实验结果.md';s=path.read_text().split('\n## 时间窗口内的最终判定')[0].rstrip();notes=[]
    for tag,title in [('second','98 图'),('third','122 图')]:
        canonical=exists('controlled_timing_'+tag)
        if canonical:z=get('controlled_timing_'+tag)
        elif tag=='third' and exists('device_timing_third'):z=get('device_timing_third')['measures']
        else:continue
        v=z['joint'];k=z['opportunity'];details=z['detail']['change'];ordinary=z['mrr']['change']
        source='原口径 CPU 完整复算' if canonical else '独立设备复算，原口径 CPU 仍在核对'
        label='controlled_timing_'+tag if canonical else 'device_timing_third'
        notes.append(f"{title}只改变 Agent 的第一个动作能否探索。固定首轮与自由首轮的双项选中率分别为 {rate(v['locked'])}、{rate(v['free'])}，配对变化 {diff(v['change'])} 个百分点。有好候选的机会从 {rate(k['locked'])} 到 {rate(k['free'])}。细节变化 {100*details['mean']:+.2f} 个百分点，普通 MRR 变化 {ordinary['mean']:+.4f}。{source}。[配对数据](results/{label}.json)")
    if exists('device_timing_third') and exists('controlled_timing_third'):
        a=get('device_timing_third');b=get('controlled_timing_third')
        gap=abs(a['measures']['joint']['change']['mean']-b['joint']['change']['mean'])*100
        notes.append(f"跨设备检查了 {len(a['checks'])} 个图像与权重的组合，图像向量最大分量差为 {max(x['max_visual'] for x in a['checks']):.2g}。局部代理名次存在数值差异，完整 122 图双项选中差值与原 CPU 口径相差 {gap:.4f} 个百分点。上文采用原 CPU 口径。[跨设备对照](results/device_timing_third.json)")
    if exists('resolution_third'):
        z=get('resolution_third')['metrics'];notes.append(f"同组原困难负例 R3 有些同分候选可进一步区分。122 图将代理分值改为固定组内倒数名次后，双项选中率从 {rate(z['original_joint'])} 到 {rate(z['ordinal_joint'])}，变化 {diff(z['change_joint'])} 个百分点。[第三批奖励并列核查](results/resolution_third.json)")
    if exists('without_gate_third'):
        z=get('without_gate_third');notes.append(f"122 图移除普通配文门槛后，选中双项改善变化 {diff(z['joint']['change'])} 个百分点，细节变化 {diff(z['detail']['change'])} 个百分点，普通 MRR 变化 {z['mrr']['change']['mean']:+.4f}。[第三批门槛对照](results/without_gate_third.json)")
    if exists('device_swap'):
        v=get('device_swap');a=v['first']['metrics']['question_minus_scene_detail'];b=v['second']['metrics']['question_minus_scene_detail']
        notes.append(f"将同场景且标注细节相反的两张图互换先验后，再比较提问对“用回本图先验”的影响。该细节优势相对原场景在两批分别变化 {diff(a)} 和 {diff(b)} 个百分点。[逐组对照](results/device_swap.json)")
    if exists('device_stochastic_seed_a_primary'):
        z=get('device_stochastic_seed_a_primary')['metrics']['CDE'];notes.append(f"另用固定权重 a 在 126 图上检验每图随机生成一次。贪心细节为 {100*z['greedy']:.2f}%，两个独立单次运行分别为 {100*z['draw1']:.2f}%、{100*z['draw2']:.2f}%，合计变化 {diff(z['change'])} 个百分点。该补充结果覆盖一个权重的细节面板。[单次随机生成](results/device_stochastic_seed_a_primary.json)")
    keys=['nomination','prefix_nomination','resolution_nomination']
    if all(exists(k) for k in keys):
        gates=[get(k) for k in keys]
        if gates[0]['selected'] is None and not gates[1]['eligible'] and not gates[2]['eligible']:
            notes.append('提问来源、首轮探索、改变并列赋分三项均未达到各自预设的 RFPO 晋级条件。本轮完成候选选择和已有权重的单次部署验证，保留此前 R2，新增 RFPO 参数更新为零。[提问来源](results/nomination.json) · [首轮探索](results/prefix_nomination.json) · [并列赋分](results/resolution_nomination.json)')
        else:notes.append(f"正式资格记录为：提问来源 {gates[0]['selected']}、首轮探索 {gates[1]['eligible']}、并列赋分 {gates[2]['eligible']}。实际更新数以时间窗口记录为准。[窗口清单](results/window.json)")
    path.write_text(s+'\n\n## 时间窗口内的最终判定\n\n'+'\n\n'.join(notes)+'\n')
if __name__=='__main__':main()
