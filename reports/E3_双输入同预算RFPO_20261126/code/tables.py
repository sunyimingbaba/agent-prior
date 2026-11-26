from study import STAGE, ARMS, CONTRASTS, read

NAMES={'retained':'现有 R2','singleplain':'只给首句，普通','singlejoint':'只给首句，联合','dualplain':'两种输入，普通','dualjoint':'两种输入，联合'}


def table(headers,rows):
    return '\n'.join('| '+' | '.join(row)+' |' for row in [headers,['---']*len(headers),*rows])


def fmt(x, metric, signed=False):
    n=x if metric=='mrr' else 100*x
    return f"{n:{'+' if signed else ''}.{4 if metric=='mrr' else 2}f}"


def interval(s,k):
    return fmt(s['mean'],k,True)+' ['+', '.join(fmt(x,k,True) for x in s['interval'])+']'


def comparison(a,b):
    return NAMES[a]+' 减 '+NAMES[b]


def main():
    reference=STAGE/'reference/ucm.json'
    z=read(reference if reference.exists() else STAGE/'results/ucm.json');x=read(STAGE/'results/transfer.json')
    txt=['# 双输入 RFPO 的单次检索与细节读数',
         '普通组只使用正确配文排名。联合组再对普通平均排名和细节同时提高的候选给额外奖励。所有百分数为两份种子等权均值，差值按百分点报告。MRR 保留原值，括号内为配对 95% 区间。']
    if reference.exists():
        txt.append('表中 UCM 的现有 R2 使用本轮同一推理环境重新生成的先验。RSITMD 沿用历史同协议的 R2。')
    for panel,title in [('hold','主要 78 图'),('repeat','上轮 64 图重复'),('shared','正式面板重合的 106 图')]:
        txt.extend(['## '+title,
            table(['策略','场景细节（%）','场景其他配文 MRR','首句细节（%）','首句其他配文 MRR'],
                  [[NAMES[a],*[fmt(z[panel+'_'+v]['levels'][a][k]['mean'],k) for v in ('scene','first') for k in ('CDE','mrr')]] for a in ARMS])])
        rows=[]
        for a,b in CONTRASTS[:3]:
            for view,label in [('scene','场景'),('first','首句')]:
                r=z[panel+'_'+view];key=a+'-'+b
                rows.append([comparison(a,b),label,interval(r['contrasts'][key]['CDE'],'CDE'),interval(r['contrasts'][key]['mrr'],'mrr'),
                             fmt(r['joint'][key]['win']['mean'],'win'),fmt(r['joint'][key]['loss']['mean'],'loss'),interval(r['joint'][key]['net'],'net')])
        txt.append(table(['比较','输入','细节差','其他配文 MRR 差','双项胜出（%）','双项失分（%）','胜出减失分'],rows))
    txt.append('## 主要 78 图的两因素交互')
    txt.append('相对只给首句的训练，双输入训练让“联合奖励减去普通奖励”的配对差增加了多少。')
    txt.append(table(['输入','细节差的变化','其他配文 MRR 差的变化'],
              [[label,*[interval(z['hold_'+v]['interaction'][k],k) for k in ('CDE','mrr')]] for v,label in [('scene','场景'),('first','首句')]]))
    txt.append('## 首句文字未区分正反细节的 56 图')
    txt.append(table(['策略','场景细节（%）','场景其他配文 MRR','首句细节（%）','首句其他配文 MRR'],
              [[NAMES[a],*[fmt(z['hold_'+v+'_unmentioned']['levels'][a][k]['mean'],k) for v in ('scene','first') for k in ('CDE','mrr')]] for a in ARMS]))
    txt.append('## UCM 两个正式终点')
    txt.append(table(['策略','210 图 mR（%）','图检文三项均值（%）','文检图三项均值（%）','126 图细节（%）'],
              [[NAMES[a],*[fmt(z['official']['levels'][a][k]['mean'],k) for k in ('mR','I2T','T2I')],fmt(z['primary']['levels'][a]['CDE']['mean'],'CDE')] for a in ARMS]))
    txt.append(table(['比较','210 图 mR 差','126 图细节差'],[[comparison(a,b),interval(z['official']['contrasts'][a+'-'+b]['mR'],'mR'),
              interval(z['primary']['contrasts'][a+'-'+b]['CDE'],'CDE')] for a,b in CONTRASTS]))
    txt.append('## RSITMD 448 图的真实首句检索')
    txt.append(table(['策略','全配文 mR（%）','排除输入配文 mR（%）'],[[NAMES[a],*[fmt(x[k]['levels'][a]['mR']['mean'],'mR') for k in ('native','alternatives')]] for a in ('retained','dualplain','dualjoint')]))
    txt.append(table(['比较','全配文 mR 差','排除输入配文 mR 差'],[[comparison(a,b),*[interval(x[k]['contrasts'][a+'-'+b]['metrics']['mR'],'mR') for k in ('native','alternatives')]] for a,b in [('dualjoint','dualplain'),('dualjoint','retained'),('dualplain','retained')]]))
    (STAGE/'实测数据表.md').write_text('\n\n'.join(txt)+'\n')


if __name__=='__main__':
    main()
