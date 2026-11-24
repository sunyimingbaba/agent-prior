import hashlib,tarfile,json,io,sys
from pathlib import Path
import numpy as np
import torch
from analyze import ROOT,sha,read,dump
def main(inputs,backup):
    p=ROOT/'pilot';index=read(p/'archives.json');counter=0;adapters=0;val_count=0
    for name,h in index['traces'].items():
        f=p/'traces'/name;assert sha(f)==h
        with tarfile.open(f) as t:
            for m in t:
                if not m.name.endswith('complete.json'):continue
                proof=json.load(t.extractfile(m));tag,subset,_=m.name.split('/')
                vec=torch.load(p/'vectors'/f'{tag}_{subset}.pt',weights_only=True)
                assert proof['adapter']==vec['adapter'] and set(proof['files'])==set(vec['images'])
                for im,src in zip(vec['images'],vec['sources']):
                    raw=t.extractfile(f'{tag}/{subset}/{im}.pt').read()
                    assert hashlib.sha256(raw).hexdigest()==proof['files'][im]==src
                    state=torch.load(io.BytesIO(raw),weights_only=True)
                    assert state['prior'].dtype==torch.bfloat16 and state['prior'].shape==(3584,) and state['image']==im and state['adapter']==proof['adapter']
                    counter+=1
    for g in ('a_R2','a_R3','b_R2','b_R3'):
        assert sha(backup/g/'adapter/adapter_model.safetensors')==read(p/g/'complete.json')['final'];adapters+=1
    for s,h in index['contexts'].items():assert sha(p/(s+'_contexts.pt'))==h
    assert sha(p/'train.json')==index['train'] and sha(ROOT/'短训口径.md')==index['pilot_protocol']
    # Independent previously validated six-recall implementation.
    sys.path.insert(0,str(ROOT.parent/'E3-2_UCM与RSITMD独立复验_20261124/code'))
    from metrics import fixed,recalls
    g=torch.load(inputs/'retrieval_text.pt',weights_only=True);dep=read(p/'deployment.json')
    labels=np.arange(210)[:,None]==np.arange(210).repeat(5)[None,:]
    errors=[]
    for tag,z in dep['levels'].items():
        x=torch.load(p/'vectors'/(tag+'_val.pt'),weights_only=True)['features']
        scores=fixed(x,g['native'])[:,g['caption_order']]
        vec=np.array([v.mean() for v in recalls(scores,labels).values()]+[v.mean() for v in recalls(scores.T,labels.T).values()])
        errors.append(float(abs(vec-z['six_recalls']).max()));val_count+=6
    assert max(errors)<1e-10 and counter==1344
    dump(p/'local_verification.json',dict(greedy_traces=counter,matched_checkpoints=adapters,verified_validation_recalls=val_count,
                 max_recall_difference=max(errors),local_checkpoint_root=str(backup)))
if __name__=='__main__':main(*(Path(v) for v in sys.argv[1:]))
