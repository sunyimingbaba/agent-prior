"""Complete the second-panel control with the same frozen reader, reusing either current-run prompt cache."""
from pathlib import Path
import os
import torch
from remote_common import ROOT,BASE,setup,table,rgb,read,sha,savept,dump
from readouts import load_model,scorer_for
@torch.inference_mode()
def main():
    setup();torch.set_num_threads(3);m=load_model(False);d=table()
    proxy=scorer_for(load_model(True),BASE/'relation_candidates_20261120/gallery/train.pt')
    seed='b';base=ROOT/'records/second'/seed;out=ROOT/'records/prefix/second'/seed
    rows=read(ROOT/'inputs/second.json');proof=read(out/'raw/complete.json')
    for j,c in enumerate(reversed(rows)):
        nm=c['image'];p=out/'raw'/(nm+'.pt');f=out/'features'/p.name
        if f.exists():continue
        raw=torch.load(p,weights_only=True);orig=torch.load(base/'raw'/p.name,weights_only=True)
        src=torch.load(base/'features'/p.name,weights_only=True)
        assert raw['rgb']==orig['rgb']==src['rgb']==c['rgb_sha256'] and raw['adapter']==orig['adapter']==src['adapter']
        assert sha(p)==proof['files'][nm] and src['source']==sha(base/'raw'/p.name)
        cache={};pix=m.preprocess(rgb(d,c))[None]
        for mode,z in orig['modes'].items():
            for h,v,r in zip(z['hidden'],src['modes'][mode]['visual'],src['modes'][mode]['ranks']):
                key=h.float().numpy().tobytes();cache[key]=(v,r)
        vs=[];rs=[]
        for h in raw['modes']['locked']['hidden']:
            h=h.float();key=h.numpy().tobytes()
            if key not in cache:
                v=m.img_embeds(pix,agent_hidden=h[None])['image_feats'][0].clone()
                assert torch.isfinite(v).all() and abs(float(v.norm())-1)<1e-4
                cache[key]=(v,int(proxy.score(h.tolist(),c['row'])[0]))
            v,r=cache[key];vs.append(v);rs.append(r)
        record=dict(image=nm,adapter=raw['adapter'],rgb=raw['rgb'],source=proof['files'][nm],
                    modes=dict(locked=dict(visual=torch.stack(vs),ranks=rs)))
        # The ordinary reader progresses in forward order. Its file takes precedence if both meet.
        tmp=f.parent/(f.name+'.secondworker');torch.save(record,tmp)
        try:os.link(tmp,f)
        except FileExistsError:
            z=torch.load(f,weights_only=True)['modes']['locked']
            assert z['ranks']==rs and torch.allclose(z['visual'],record['modes']['locked']['visual'],atol=1e-5,rtol=0)
        finally:tmp.unlink()
        if (j+1)%12==0:print(j+1,len(rows),flush=True)
    for sh in (0,1):
        files={}
        for c in rows[sh::2]:
            p=out/'features'/(c['image']+'.pt');z=torch.load(p,weights_only=True)
            assert z['source']==proof['files'][c['image']];files[c['image']]=sha(p)
        path=out/'features'/('complete_'+str(sh)+'.json')
        if path.exists():assert read(path)['files']==files
        else:dump(path,dict(files=files))
if __name__=='__main__':main()
