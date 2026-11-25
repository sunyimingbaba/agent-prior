"""Re-read both candidate archives with the historical frozen CUP on CPU."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
PREV=ROOT.parent/'multicap_rfpo_20261124'
sys.path.insert(0,str(PREV/'code'))
from runtime import pool,table,rgb,setup,sha,read,savept,dump
from readouts import load_model

@torch.inference_mode()
def main():
    setup();cup=load_model(False);cases=pool();t=table();manifest={}
    for batch,folder in [('development','attribute_questions_20261125'),('repeat','attribute_questions_repeat_20261125')]:
        src=ROOT.parent/folder;lock=read(src/'inputs/lock.json')
        for seed in 'ab':
            store=src/'records'/seed;cp=read(store/'complete.json')
            out={}
            for row in lock['rows']:
                nm=row['image'];f=store/(nm+'.json');p=store/(nm+'.pt')
                assert sha(f)==cp['records'][nm] and sha(p)==read(f)['states']
                image=cup.preprocess(rgb(t,cases[nm]))[None];cache={};v={}
                for mode,states in torch.load(p,weights_only=True).items():
                    features=[]
                    for h in states:
                        x=h.float();key=x.numpy().tobytes()
                        if key not in cache:cache[key]=cup.img_embeds(image,agent_hidden=x[None])['image_feats'][0].cpu().clone()
                        features.append(cache[key])
                    v[mode]=torch.stack(features)
                    assert v[mode].shape[0]==17 and torch.isfinite(v[mode]).all()
                out[nm]=v
            target=ROOT/'records'/(batch+'_'+seed+'.pt');savept(target,out)
            manifest[target.name]=dict(sha256=sha(target),adapter=cp['adapter'],lock=cp['lock'],records=sha(store/'complete.json'))
            print(batch,seed,len(out),flush=True)
    dump(ROOT/'records/features.json',manifest)
if __name__=='__main__':main()
