"""Render verified measurements without hand-transcribing percentages or confidence intervals."""
from rosters import ROOT,read
def r(n):return read(ROOT/'results'/(n+'.json'))
def f(x):return f'{100*x:.2f}%'
def d(z):
    lo,hi=z['interval']
    return f"{100*z['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"
def m(z):return f(z['mean'])
def table(cols,rows):
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(map(str,z))+' |' for z in rows])
def report():
    rows=[]
    for t in ('first','second'):
        a=r(t);v=a['by_pair']
        for mode,name in [('scene','原场景 16 条'),('mixed','各 8 条'),('question','全提问 16 条')]:
            rows.append([str(a['images']),name,m(v['opportunity'][mode]),m(v['joint'][mode]),m(v['hit_given_opportunity'][mode]),f"{v['mrr'][mode]['mean']:.4f}",m(v['detail'][mode])])
    sections=['## 同一张图的 16 个候选怎么生成',table(['纳入图数','来源','有双项改善候选','R3 选中双项改善','有机会时选中','留出配文 MRR','细节正确率'],rows)]
    rows=[]
    for t in ('first','second'):
        z=r('historical_timing_'+t)['joint'];rows.append([t,m(z['locked']),m(z['free']),d(z['change'])])
    sections+=['## 原场景历史设置比较',table(['图组','旧设置','全轨迹探索','双项选中变化（百分点）'],rows)]
    if all((ROOT/'results'/('controlled_timing_'+t+'.json')).exists() for t in ('second','third')):
        rows=[]
        for t in ('second','third'):
            z=r('controlled_timing_'+t)
            for k,name in [('joint','双项选中'),('opportunity','有双项改善候选'),('detail','细节正确率'),('mrr','未知配文倒数名次 ×100')]:
                v=z[k];rows.append([str(z['images']),name,m(v['locked']),m(v['free']),d(v['change'])])
        sections+=['## 只改变首轮能否探索',table(['纳入图数','指标','固定首轮','自由首轮','配对差及 95% 区间（百分点或 MRR×100）'],rows)]
    rows=[]
    for t in ('first','second','third'):
        if not (ROOT/'results'/('resolution_'+t+'.json')).exists():continue
        z=r('resolution_'+t);v=z['metrics'];rows.append([z['images'],m(v['distinct_winners_pruned']),m(v['original_joint']),m(v['ordinal_joint']),d(v['change_joint']),d(v['change_detail'])])
    sections+=['## 困难负例奖励的并列分值',table(['图数','能进一步区分赢家','原 R3 双项选中','新赋分双项选中','选中变化及区间','细节变化及区间'],rows)]
    rows=[]
    for t in ('second','third'):
        if not (ROOT/'results'/('attribution_'+t+'.json')).exists():continue
        z=r('attribution_'+t);v=z['all']['metrics'];lev=z['levels']['free']
        rows.append([z['all']['images'],f(lev['16_random_joint']),f(lev['16_R2_joint']),f(lev['16_R3_joint']),d(v['joint']['free_R3_minus_R2']),d(v['joint']['free_16_minus_8_R3'])])
    if rows:sections+=['## 同组随机选、R2 选、困难负例 R3 选',table(['图数','随机选中双项改善','R2','R3','R3 减 R2','R3 十六条减八条'],rows)]
    if (ROOT/'results/route.json').exists():
        v=r('route');rows=[]
        for sub,metric,label in [('primary','CDE','126 图细节'),('official','mR','210 图普通检索')]:
            rows.append([label,m(v[sub]['baseline'][metric]),m(v[sub]['route'][metric]),d(v[sub]['change'][metric])])
        sections+=['## 原图只生成一份先验，是否增加自动提问',table(['测试','原先验','自动提问','配对差及 95% 区间（百分点）'],rows)]
    rows=[]
    for t in ('ucm','rsitmd'):
        if not (ROOT/'results'/('single_'+t+'.json')).exists():continue
        for mode,label in [('native','全部配文'),('alternatives','不同于输入首句')]:
            v=r('single_'+t)[mode];rows.append([t.upper(),label,v['eligible'],v['captions'],m(v['baseline']['mR']),m(v['route']['mR']),d(v['change']['mR'])])
    if rows:sections+=['## 自动提问的完整图库与非输入配文',table(['数据集','文本','图数','配文数','原 mR','提问 mR','配对差及区间'],rows)]
    if (ROOT/'results/stochastic_single.json').exists():
        z=r('stochastic_single');rows=[]
        for sub,k in [('primary','CDE'),('official','mR')]:
            v=z[sub]['metrics'];rows.append([sub,m(v['greedy'][k]),m(v['draw1'][k]),m(v['draw2'][k]),m(v['average'][k]),d(v['change'][k])])
        sections+=['## 每图独立随机生成一份先验',table(['面板','原贪心','随机轮次 1','随机轮次 2','两轮平均','随机减贪心及区间'],rows)]
    for kind in ('prefix','resolution'):
        if not (ROOT/'results'/(kind+'_learned.json')).exists():continue
        z=r(kind+'_learned');rows=[]
        for arm in sorted(set(z['primary']['levels'])|set(z['official']['levels'])):
            row=[arm]
            for sub,k in [('primary','CDE'),('official','mR')]:
                v=z[sub]['levels'].get(arm,{}).get(k)
                row.append(m(v)+'（'+','.join(v['seeds'])+'）' if v else '待完成')
            rows.append(row)
        sections+=['## 从 R0 短训后的实际单次部署，'+('首轮探索' if kind=='prefix' else '并列赋分'),table(['训练组与覆盖种子','126 图细节','210 图 mR'],rows)]
    out=ROOT/'逐项数值.md'
    out.write_text('# E3 两小时实验逐项数值\n\n区间按照预定的图或自然图对计算。所有“选中双项改善”均要求普通留出配文倒数名次和细节正确率同时严格超过同图原贪心。\n\n'+'\n\n'.join(sections)+'\n')
if __name__=='__main__':report()
