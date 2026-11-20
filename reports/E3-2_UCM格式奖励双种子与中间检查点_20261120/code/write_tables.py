"""Export compact, filterable tables in the units used in the experiment report."""
from shared import ROOT,read
from metrics import csvwrite
from analyze import study_states,E1SETS

def fmt(x,n=3):
    return round(float(x),n)

def main():
    r=ROOT/'results';states=read(r/'states.json');contrasts=read(r/'paired_contrasts.json')
    history=[]; interventions=[]; e1=[]
    for state in study_states():
        z=states[state];native=z['native'];rel=z['detail']
        seed='共享起点' if state=='base_00' else state[0].upper()
        arm='历史权重' if state=='base_00' else '保留格式分' if '_on_' in state else '无格式分'
        step=0 if state=='base_00' else int(state.split('_')[2])
        forced=state.endswith('_force')
        if not forced:
            history.append({'种子':seed,'条件':arm,'步数':step,
                       '原首句mR(%)':fmt(native['mR']*100),'通用查询mR(%)':fmt(z['generic']['mR']*100),
                       '第一步结束图数/210':native['eos'],'使用工具图数/210':native['tools'],'完整答案标签图数/210':native['formatted'],
                       '平均生成token':fmt(native['mean_tokens']),'双视图均对题数/32':rel['both'],'单视图PairAcc(%)':fmt(rel['pairacc']*100),
                       '方向题平均余弦分差':fmt(rel['margin'],7)})
        else:
            normal_state=state.removesuffix('_force');normal=states[normal_state];pair=contrasts[f'{state[0]}_{state.split("_")[1]}_force-natural']['native_mR']
            interventions.append({'种子':seed,'训练条件':arm,'自然结束数/210':normal['native']['eos'],
                     '屏蔽首步结束后结束数/210':native['eos'],'自然mR(%)':fmt(normal['native']['mR']*100),'干预mR(%)':fmt(native['mR']*100),
                     'mR差(百分点)':fmt(pair['mean']*100),'95%下限(百分点)':fmt(pair['lo']*100),'95%上限(百分点)':fmt(pair['hi']*100),
                     '自然双视图均对/32':normal['detail']['both'],'干预双视图均对/32':rel['both'],'自然工具图数/210':normal['native']['tools'],'干预工具图数/210':native['tools']})
        for part in ('native','generic'):
            if part not in z:continue
            for group in E1SETS:
                x=z[part]['e1'][group]
                e1.append({'种子':seed,'条件':arm,'步数':step,'解码':'屏蔽首步结束' if forced else '自然','先验查询':'原首句' if part=='native' else '通用',
                           '分项':{'quantity_A':'数量A','quantity_B':'数量B','attribute':'属性','relation':'关系'}[group],'判对数':x['correct'],
                           '判断总数':x['n'],'PairAcc(%)':fmt(x['correct']/x['n']*100),'平均余弦分差':fmt(x['margin'],7),
                           '反向实拍两图都对/18':x.get('both','')})
    csvwrite(r/'检查点汇总.csv',history);csvwrite(r/'屏蔽首步结束符.csv',interventions);csvwrite(r/'E1分项汇总.csv',e1)
if __name__=='__main__':main()
