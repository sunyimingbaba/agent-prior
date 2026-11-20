"""Collect the exact deployed endpoints, E1 subgroups and sampled credit into compact tables."""
import csv
from collections import defaultdict
from shared import ROOT, read, dump
from metrics import csvwrite
from analyze import E1SETS

def main():
    out=ROOT/'results'
    states=read(out/'states.json')
    turns=read(out/'quarter_stats.json')['states']
    compact=[];subgroups=[]
    for state,row in states.items():
        detail,panel=row['detail'],row['trainpanel']
        add=turns.get(state,{})
        compact.append(dict(策略=state,原首句mR=row['native']['mR'],通用查询mR=row['generic']['mR'],
                            原图与半转均对=detail['both'],顺逆四分之一转均对=add.get('quarter_both',''),
                            四朝向均对=add.get('all_four',''),验证原图单侧准确率=detail['upright'],
                            验证半转单侧准确率=detail['rotated'],验证平均较弱侧分差=detail['weakest'],
                            训练表述A均对=panel['both_by_word']['A'],训练保留表述B均对=panel['both_by_word']['B'],
                            原首句首步结束图数=row['native']['eos'],原首句调用工具图数=row['native']['tools']))
        for part in ('native','generic'):
            for key in E1SETS:
                one=row[part]['e1'][key]
                subgroups.append(dict(策略=state,提示=part,项目=key,正确=one['correct'],总题数=one['n'],
                                      平均分差=one['margin'],反向两图均对=one.get('both','')))
    csvwrite(out/'检查点汇总.csv',compact)
    csvwrite(out/'E1分项汇总.csv',subgroups)
    by=defaultdict(list)
    with (out/'same_candidate_credit.csv').open(encoding='utf-8-sig') as fp:
        for row in csv.DictReader(fp):by[row['arm']].append(row)
    aggregated={}
    for arm,rows in by.items():
        counts={key:sum(int(r[key]) for r in rows) for key in ('paired_both','upright_correct','rotated_correct',
               'all_pairings_min','all_pairings_max','any_possible','changed_signs','top_set_changed_views','advantage_changed')}
        counts.update(candidate_pairs=4*len(rows),views=2*len(rows),groups=len(rows))
        aggregated[arm]=counts
    dump(out/'credit_summary.json',aggregated)
if __name__=='__main__':
    main()
