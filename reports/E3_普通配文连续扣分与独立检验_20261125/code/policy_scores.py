import numpy as np
from compare import OUT,read,load,sha,dump
from single_scores import FORM,RFPO,calculate,summary_by_seed
from locked_selector import ordinal_prediction
from orders import score,advantages,calibrate
from gate import K
def main():
    lock=read(OUT/'inputs/multi_nomination.json');rec=read(OUT/'results/real_confirmation.json');assert rec['eligible']
    names=['R2','old','new','untrained','retained'];levels={};summary={};audit={}
    for tag,rows in [('primary',read(RFPO/'inputs/panel.json')),('official',read(FORM/'inputs/ucm.json'))]:
        scores={a:{} for a in names}
        for s in 'ab':
            for arm in names:
                if arm in ('untrained','retained'):
                    obj=load(RFPO/'vectors'/(s+'_'+('R0' if arm=='untrained' else 'deployed')+'_'+tag+'.pt'))
                else:
                    dest=OUT/'policy'/(s+'_'+arm);cp=read(dest/'complete.json');steps=read(dest/'steps.json')
                    assert len(steps)==cp['steps']==24 and cp['logs']==sha(dest/'steps.json')
                    assert cp['receipt']==sha(OUT/'results/real_confirmation.json')
                    for z in steps:
                        u=z['scores'];o=ordinal_prediction(lock['model'],lock['weights'],np.array(u['features']));assert np.allclose(o,u['order'],atol=1e-10)
                        rw,_=score([{k:u[k][i] for k in ('rank','native','A','keep')} for i in range(16)],K)
                        expected={'R2':rw['R2'],'old':rw['R3'],'new':calibrate(o,u['rank'][:16])}[arm]
                        assert np.allclose(expected,z['rewards'][arm],atol=1e-12) and np.allclose(advantages(expected),z['adv'],atol=1e-8)
                    obj=load(dest/'evaluate'/(tag+'.pt'));assert obj['adapter']==cp['final']
                    audit[s+'_'+arm]=dict(steps=cp['steps'],updated=cp['updated'],source=cp['initial'],final=cp['final'])
                assert obj['images']==[r['image'] for r in rows];scores[arm][s]=calculate(tag,obj['features'],rows)
        labs=[r.get('pair',r['image']) for r in rows];levels={a:summary_by_seed(u,labs) for a,u in scores.items()}
        comp={a:summary_by_seed({s:{k:x-scores[a][s][k] for k,x in scores['new'][s].items()} for s in 'ab'},labs) for a in names if a!='new'}
        summary[tag]=dict(levels=levels,changes=comp)
    dump(OUT/'results/policy_results.json',dict(audit=audit,**summary))
if __name__=='__main__':main()
