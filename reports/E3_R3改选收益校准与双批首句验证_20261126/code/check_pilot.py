from data import ROOT,read,stream,previous_panel,previous_banks,OLD,PREV
import numpy as np
import joblib
import torch
from online_reward import Scorer
from neighbor import local_packets,prototypes,old_choices,decide,gain_study

def main():
    train=read(ROOT/'inputs/pilot/train.json');ro={r['image']:r for r in read(OLD/'inputs/real_roster.json')}
    full=list(stream('dev'));pool=prototypes(full)
    model=joblib.load(ROOT/'inputs/switch_model.joblib');nom=read(ROOT/'inputs/提名.json');loc=read(ROOT/'inputs/配文预测提名.json')
    old=read(PREV/'锁定评分参数.json')['model'];w=np.array(old['coef'])/np.array(old['scale_x'])[:,None]
    total=0;difference=0
    for s in 'ab':
        sc=Scorer(s);wanted={r['image'] for r in train[s]}
        sub=[row for row in full if row[0]['image'] in wanted and row[1]==s and row[2]=='first']
        assert len(sub)==24
        for row in sub:
            for packet in gain_study.packets([row]):
                # Check averaged predictions of both implementations; matched tops include every tied candidate.
                hard=model['trees'].predict(packet['x'].reshape(-1,packet['x'].shape[-1]))
                portable=sc.tree_prediction(packet['x'].reshape(-1,packet['x'].shape[-1]))
                difference=max(difference,float(np.max(abs(hard-portable))))
                oo=packet['old']@w;tri=old_choices(oo.mean(0),packet)
                pick=gain_study.choose(packet,hard.reshape(3,8,4).mean(0),tri,nom['weight'],nom['mode'])
                start=8*packet['block'];online=sc.run(packet['image'],row[-1][start:start+8],packet['held'])
                assert online['switch']==pick and online['old3']==tri and online['ordinary']==packet['ref']
                assert online['old']==old_choices(oo[0],packet)
                total+=1
            for p in local_packets([row],pool,set()):
                rank=p['local'][loc['k']]@loc['coefficients'][str(loc['k'])]
                tested=decide(p,(p['old']@w).mean(0),rank,loc['weight'],loc['mode'])
                start=8*p['block'];assert sc.run(p['image'],row[-1][start:start+8],p['held'])['neighbor']==tested
    assert difference<1e-12
    (ROOT/'results/online_parity.json').write_text(__import__('json').dumps(dict(combinations=total,max_tree_error=difference),indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(1)
    main()
