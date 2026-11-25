import time
import sys
import torch
from remote_common import BASE,ROOT,roster,directory,sha,read,table,rgb,setup,savept,dump
from readouts import load_model,scorer_for

@torch.inference_mode()
def main(shard):
    setup();torch.set_num_threads(3);data=table()
    model=load_model(False);scorer=scorer_for(load_model(True),BASE/'relation_candidates_20261120/gallery/train.pt')
    for tag in ('first','second'):
        for seed in 'ab':
            src=directory(tag,seed,'raw');dst=directory(tag,seed,'features')
            rows=roster(tag)[shard::2]
            for j,row in enumerate(rows):
                nm=row['image'];f=src/(nm+'.pt');out=dst/f.name
                while not f.exists():time.sleep(1)
                if out.is_file():
                    assert torch.load(out,weights_only=True)['source']==sha(f)
                    continue
                obj=torch.load(f,weights_only=True)
                assert obj['rgb']==row['rgb_sha256'] and obj['image']==nm
                pix=model.preprocess(rgb(data,row))[None];cache={}
                result=dict(source=sha(f),adapter=obj['adapter'],rgb=obj['rgb'],image=nm,modes={})
                for mode,raw in obj['modes'].items():
                    v=[];ranks=[]
                    for state in raw['hidden']:
                        h=state.float();key=h.numpy().tobytes()
                        if key not in cache:
                            feat=model.img_embeds(pix,agent_hidden=h[None])['image_feats'][0]
                            assert torch.isfinite(feat).all() and abs(feat.norm().item()-1)<1e-4
                            cache[key]=(feat.clone(),int(scorer.score(h.tolist(),row['row'])[0]))
                        ft,rk=cache[key];v.append(ft);ranks.append(rk)
                    result['modes'][mode]=dict(visual=torch.stack(v),ranks=ranks)
                savept(out,result)
                print(tag,seed,j+1,len(rows),len(cache),flush=True)
            dump(dst/('complete_'+str(shard)+'.json'),dict(lock=sha(ROOT/'inputs/lock.json'),files={r['image']:sha(dst/(r['image']+'.pt')) for r in rows}))
if __name__=='__main__':main(int(sys.argv[1]))
