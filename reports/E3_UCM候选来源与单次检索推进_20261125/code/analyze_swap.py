import sys
import numpy as np
from rosters import ROOT,OLD,PRIOR,read,dump,sha
from report_scores import bank,load,greedy_score
from run_stage import boot

def main():
    for tag in ('first','second'):
        panel=read(ROOT/'inputs'/(tag+'.json'));b=bank(OLD if tag=='first' else PRIOR/'reward_confirm')
        rows=[]
        for seed in 'ab':
            base=ROOT/'records'/tag/seed
            swap=load(base/'swapped.pt');assert swap['proof']==sha(base/'raw/complete.json')
            assert swap['protocol']==sha(ROOT/'同图与配对图先验核查口径.md')
            for c in panel:
                nm=c['image']
                if len(set(b['maps'][nm]['native']))<3:continue
                x=load(base/'features'/(nm+'.pt'));sw=swap['conditions'][nm]
                assert x['adapter']==swap['adapter']
                vals={}
                for mode in ('scene','question'):
                    vals['own_'+mode]=greedy_score(b,c,x['modes'][mode]['visual'][-1].double().numpy())
                    vals['peer_'+mode]=greedy_score(b,c,sw['visual'][mode].double().numpy())
                rows.append(dict(image=nm,peer=sw['peer'],pair=c['pair'],seed=seed,**vals))
        result={}
        label=[r['pair'] for r in rows]
        for k in ('detail','mrr'):
            v={a:np.array([z[a][k] for z in rows]) for a in ('own_scene','own_question','peer_scene','peer_question')}
            v['scene_match_gain']=v['own_scene']-v['peer_scene']
            v['question_match_gain']=v['own_question']-v['peer_question']
            v['difference_of_gains']=v['question_match_gain']-v['scene_match_gain']
            result[k]={m:boot(x,np.ones(len(x)),label) for m,x in v.items()}
        dump(ROOT/'results'/('swap_'+tag+'.json'),result)
        dump(ROOT/'records'/('swap_'+tag+'_images.json'),rows)
if __name__=='__main__':main()
