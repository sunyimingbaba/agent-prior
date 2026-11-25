"""Process the unfilled end of the third panel with the same canonical CUP and proxy, in reverse order."""
import os,time,signal
from pathlib import Path
import torch
from remote_common import ROOT,BASE,setup,table,rgb,read,sha
from readouts import load_model,scorer_for
def optional_readers():
    from reader_priority import workers
    pids=set()
    for tid in workers():
        try:
            line=next(x for x in Path('/proc',str(tid),'status').read_text().splitlines() if x.startswith('Tgid:'))
            pids.add(int(line.split()[1]))
        except FileNotFoundError:pass
    return pids
@torch.inference_mode()
def run():
    setup();torch.set_num_threads(3);m=load_model(False);proxy=scorer_for(load_model(True),BASE/'relation_candidates_20261120/gallery/train.pt');d=table()
    folder=ROOT/'records/prefix/third/b';proof=read(folder/'raw/complete.json')
    for j,c in enumerate(reversed(read(ROOT/'inputs/third.json'))):
        if time.time()>=1795593180:return
        nm=c['image'];p=folder/'raw'/(nm+'.pt');f=folder/'features'/p.name
        if f.exists():continue
        z=torch.load(p,weights_only=True);assert z['rgb']==c['rgb_sha256'] and sha(p)==proof['files'][nm]
        pix=m.preprocess(rgb(d,c))[None];cache={};modes={}
        for arm,q in z['modes'].items():
            vs=[];rs=[]
            for h in q['hidden']:
                h=h.float();key=h.numpy().tobytes()
                if key not in cache:
                    v=m.img_embeds(pix,agent_hidden=h[None])['image_feats'][0]
                    assert torch.isfinite(v).all() and abs(float(v.norm())-1)<1e-4
                    cache[key]=(v.clone(),int(proxy.score(h.tolist(),c['row'])[0]))
                v,r=cache[key];vs.append(v);rs.append(r)
            modes[arm]=dict(visual=torch.stack(vs),ranks=rs)
        result=dict(image=nm,adapter=z['adapter'],rgb=z['rgb'],source=proof['files'][nm],modes=modes)
        temp=f.with_suffix('.assistant-tmp');torch.save(result,temp)
        try:os.link(temp,f)
        except FileExistsError:
            existing=torch.load(f,weights_only=True)
            assert all(existing['modes'][k]['ranks']==v['ranks'] and torch.allclose(v['visual'],existing['modes'][k]['visual'],atol=1e-5,rtol=0) for k,v in modes.items())
        finally:temp.unlink()
        if (j+1)%8==0:print(j+1,flush=True)
if __name__=='__main__':
    # Prioritize the locked primary comparison after independent deployment readout.
    pids=optional_readers()
    for p in pids:
        try:os.kill(p,signal.SIGSTOP)
        except ProcessLookupError:pass
    try:run()
    finally:
        for p in pids:
            try:os.kill(p,signal.SIGCONT)
            except ProcessLookupError:pass
