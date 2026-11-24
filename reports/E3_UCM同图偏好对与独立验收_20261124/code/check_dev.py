"""Cross-check recall ties and detail votes with the previous independent UCM scoring implementation."""
import sys
import numpy as np
from dev_metrics import ROOT, HIST, REP, BASE, read, load, dump
sys.path.insert(0,str(REP/'E3-2_UCM与RSITMD独立复验_20261124/code'))
from metrics import fixed, recalls

def check():
    levels={}
    for label in ('observed','continuous','bounded'):
        levels.update(read(ROOT/'results'/('dev_'+label+'.json'))['levels'])
    val=load(BASE/'retrieval_text.pt'); panel=read(HIST/'inputs/panel.json')
    bank=load(HIST/'records/fresh_text.pt')
    errors=[]
    for tag,expected in levels.items():
        root=HIST if tag.endswith('_start') else ROOT
        visual=load(root/'pilot/vectors'/(tag+'_val.pt'))['features']
        sim=fixed(visual,val['native'])[:,val['caption_order']]
        mask=np.arange(len(visual))[:,None]==np.repeat(np.arange(len(visual)),5)[None,:]
        score=[x.mean() for x in recalls(sim,mask).values()]+[x.mean() for x in recalls(sim.T,mask.T).values()]
        errors.extend(abs(np.array(score)-expected['recalls']).tolist())
        fresh=load(root/'pilot/vectors'/(tag+'_fresh.pt'))
        s=fixed(fresh['features'],bank['hard'])
        for k in 'CDE':
            cols=np.array([bank['maps'][c['image']][k] for c in panel]);i=np.arange(len(cols))
            assert (s[i,cols[:,0]]>s[i,cols[:,1]]).mean()==expected[k]
    assert max(errors)<1e-10
    dump(ROOT/'results/check_dev.json',dict(arms=len(levels),recalls=len(errors),max_abs_error=max(errors),detail_scores=3*len(levels)))
if __name__=='__main__':check()
