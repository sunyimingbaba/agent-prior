"""Verify transfer digests, individual priors, deployment text readout and independently recompute recalls."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
import numpy as np,torch
from score import ROOT,REP,OLD,ARMS,BASE,read,sha,dump,load
sys.path.insert(0,str(REP/'E3-2_UCM与RSITMD独立复验_20261124/code'))
from metrics import fixed,recalls
def main(backup):
    pil=ROOT/'pilot';idx=read(pil/'archives.json');dep=read(ROOT/'results/deployment.json')
    assert idx['protocol']==sha(ROOT/'实验口径.md') and idx['rosters']==sha(ROOT/'inputs/rosters.json') and idx['screen']==sha(ROOT/'screen/records/extra_provenance.json')
    for key,file in (('tools','tools.json'),('fresh_text','fresh_text.pt')):assert idx[key]==sha(ROOT/'records'/file)
    assert idx['vectors']==read(pil/'vectors/index.json');assert len(idx['traces'])==10
    for s,h in idx['contexts'].items():assert sha(ROOT/'records'/(s+'_contexts.pt'))==h
    tools=read(ROOT/'records/tools.json')
    cf=read(ROOT/'inputs/panel.json');vl=read(BASE/'validation.json');lookup={k:{r['image']:r for r in rows} for k,rows in [('fresh',cf),('val',vl)]}
    total=0;usage={};adapters={}
    for name,digest in idx['traces'].items():
        f=pil/'traces'/name;assert sha(f)==digest
        with tarfile.open(f) as t:
            for m in t:
                if not m.name.endswith('/complete.json'):continue
                proof=json.load(t.extractfile(m));tag,sub,_=m.name.split('/');vec=load(pil/'vectors'/f'{tag}_{sub}.pt')
                assert vec['adapter']==proof['adapter'] and set(vec['images'])==set(proof['files'])==set(lookup[sub])
                calls=[];second=[]
                for nm,h in zip(vec['images'],vec['sources'],strict=True):
                    raw=t.extractfile(f'{tag}/{sub}/{nm}.pt').read();assert hashlib.sha256(raw).hexdigest()==proof['files'][nm]==h
                    r=torch.load(io.BytesIO(raw),weights_only=True);c=lookup[sub][nm]
                    assert r['rgb']==c['rgb_sha256'] and r['query']==c['scene_query' if sub=='fresh' else 'orig_query'] and r['image']==nm
                    assert r['prior'].shape==(3584,) and r['prior'].dtype==torch.bfloat16 and torch.isfinite(r['prior']).all() and r['adapter']==vec['adapter']
                    assert 1<=len(r['rounds'])<=3 and len(r['rounds'])-1<=len(r['tools'])<=3
                    if sub=='fresh':assert all(answer==tools[nm][0] for answer in r['tools'])
                    calls.append(len(r['tools']));second.append(len(r['rounds'])>1);total+=1
                usage[tag+'_'+sub]=dict(n=len(calls),called=sum(x>0 for x in calls),mean=float(np.mean(calls)),second_round=sum(second))
    assert total==len(cf)*10+len(vl)*8
    for s in ('a','b'):
        oldval=load(OLD/'pilot/vectors'/(s+'_start_val.pt'));current=load(pil/'vectors'/(s+'_start_val.pt'))
        assert torch.equal(current['features'],oldval['features']) and current['sources']==oldval['sources'] and current['adapter']==oldval['adapter']
        for a in ARMS:
            tag=s+'_'+a;h=sha(backup/tag/'adapter/adapter_model.safetensors');assert h==read(pil/tag/'complete.json')['final']
            adapters[tag]=h
    g=load(BASE/'retrieval_text.pt');bank=load(ROOT/'records/fresh_text.pt')
    mask=np.arange(len(vl))[:,None]==np.arange(len(vl)).repeat(5)[None,:];errors=[]
    for tag,metrics in dep['levels'].items():
        v=load(pil/'vectors'/(tag+'_val.pt'));sim=fixed(v['features'],g['native'])[:,g['caption_order']]
        vec=[x.mean() for x in recalls(sim,mask).values()]+[x.mean() for x in recalls(sim.T,mask.T).values()]
        errors.append(float(np.max(abs(np.array(vec)-metrics['six_recalls']))))
        f=load(pil/'vectors'/(tag+'_fresh.pt'));fs=fixed(f['features'],bank['hard'])
        for l in 'CDE':
            ok=[arr[k[0]]>arr[k[1]] for arr,k in zip(fs,[bank['maps'][r['image']][l] for r in cf])]
            assert np.mean(ok)==metrics[l]
    assert max(errors)<1e-10
    dump(ROOT/'results/verification.json',dict(traces=total,matched_checkpoints=adapters,recalls_checked=60,max_abs_error=max(errors),fresh_wording_levels_checked=30,tool_usage=usage))
if __name__=='__main__':main(Path(sys.argv[1]))
