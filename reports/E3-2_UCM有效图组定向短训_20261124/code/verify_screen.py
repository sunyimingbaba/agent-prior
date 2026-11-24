"""Independent eligibility check from both batches of saved candidates, scores and manifests."""
import hashlib,io,tarfile
from collections import Counter
from pathlib import Path
import numpy as np,torch
from score import ROOT,REP,OLD,read,sha,dump,load
def unpack(folder,tag):
    archive=folder/(tag+'.tar.gz');prov=read(folder/('extra_provenance.json' if tag.startswith('extra') else 'provenance.json'))
    assert sha(archive)==prov['archives'][tag]
    data={}
    with tarfile.open(archive) as tf:
        def raw(p):return tf.extractfile(p).read()
        import json
        ps={k:json.loads(raw(k+'/complete.json')) for k in (tag,tag+'_readouts')}
        for nm,h in ps[tag]['files'].items():
            samples=[]
            for k in (tag,tag+'_readouts'):
                binary=raw(k+'/'+nm+'.pt');assert hashlib.sha256(binary).hexdigest()==ps[k]['files'][nm]
                samples.append(torch.load(io.BytesIO(binary),weights_only=True))
            s,v=samples;assert v['source']==h and v['post_tool']==s['post_tool'] and s['adapter']==ps[tag]['adapter']==ps[tag+'_readouts']['adapter']
            data[nm]=(s,v)
    return data
def main():
    torch.set_num_threads(2);ro=read(ROOT/'inputs/rosters.json');summary={}
    panel=read(ROOT/'inputs/panel.json');assert len(panel)==2*read(ROOT/'inputs/lock.json')['pairs']
    for seed in ('a','b'):
        audit=[];pids={};count=0
        for sub in ('','confirm'):
            prefix=OLD/sub;panel0=read(prefix/'inputs/panel.json')
            path=prefix/'records';bank=load(path/'text.pt');check=read(path/'provenance.json');assert sha(path/'text.pt')==check['text']
            src=ROOT/'screen/records' if sub else path;px=read(src/'extra_provenance.json');assert px['text']==check['text'] and px['tools']==check['tools'] and px['adapters']==check['adapters']
            first,extra=unpack(path,seed),unpack(src,'extra_'+seed)
            for c in panel0:
                n=c['image'];a,x=first[n];b,y=extra[n];post=a['post_tool'];pids[n]=c
                assert a['rgb']==b['rgb']==c['rgb_sha256'] and a['first']==b['first'] and a['tool_calls']==b['tool_calls'] and post==b['post_tool']
                assert a['seed']!=b['seed'] and a['adapter']==b['adapter']
                vals=torch.cat((x['visual'],y['visual'])).double().numpy();m=bank['maps'][n];h=bank['hard'].double().numpy()
                sa=vals@(h[m['A'][0]]-h[m['A'][1]]);sb=vals@(h[m['B'][0]]-h[m['B'][1]])
                pos=int(((sa>0)&(sb>0)).sum());neg=int(((sa<0)&(sb<0)).sum())
                audit.append(dict(image=n,post=post,both_correct=pos,both_wrong=neg));count+=len(sa)
        assert audit==ro[seed]['selection_counts'] and count==2688
        e=[r['image'] for r in audit if r['post'] and r['both_correct'] and r['both_wrong']]
        assert e==ro[seed]['eligible'] and sum(r['post'] for r in audit)==ro[seed]['available']
        assert not (set(pids)&{r['image'] for r in panel})
        r=ro[seed];summary[seed]=dict(images=len(audit),candidate_slots=count,post_tool=r['available'],eligible=len(e),used_selected=len(set(r['selected'])),used_regular=len(set(r['regular'])),control_overlap=len(set(r['selected'])&set(r['regular'])),control_eligible_visits=sum(x in e for x in r['regular']),selected_eligible_visits=sum(x in e for x in r['selected']),selected_scene_counts=dict(Counter(pids[x]['scene'] for x in set(r['selected']))))
    dump(ROOT/'results/screen_verification.json',summary)
if __name__=='__main__':main()
