"""Finish the paired-image swap readout using saved trajectories and compare against canonical CPU outputs."""
from collections import defaultdict
import hashlib,io
import numpy as np
import torch
from PIL import Image
from device_readout import ROOT,read,sha,dump,load,load_device,arrow
from when_to_sample import archive_raw,OLD,PRIOR,bank
from report_scores import greedy_score
from run_stage import boot
@torch.inference_mode()
def main():
    model,cv,_=load_device();t=arrow('train');out={}
    for tag in ('first','second'):
        rows=read(ROOT/'inputs'/(tag+'.json'));pairs=defaultdict(list)
        for c in rows:pairs[c['pair']].append(c['image'])
        b=bank(OLD if tag=='first' else PRIOR/'reward_confirm');metrics=[];checks=[]
        for s in 'ab':
            folder=ROOT/'records'/tag/s;fp=folder/'swapped.pt';canonical=load(fp) if fp.exists() else None
            raw={z['image']:z for z in archive_raw(folder,rows)}
            for i,c in enumerate(rows):
                nm=c['image']
                if len(set(b['maps'][nm]['native']))<3:continue
                peer=next(n for n in pairs[c['pair']] if n!=nm);f=load(folder/'features'/(nm+'.pt'))['modes']
                if canonical is None or i%32==0:
                    rec=raw[peer];im=Image.open(io.BytesIO(t['image'][c['row']].as_py())).convert('RGB')
                    assert hashlib.sha256(im.tobytes()).hexdigest()==c['rgb_sha256']
                    h=torch.stack([rec['modes'][k]['hidden'][-1] for k in ('scene','question')]).float().to('mps')
                    vals=model(cv(im)[None].repeat(2,1,1,1).to('mps'),h).cpu()
                    alt=dict(zip(('scene','question'),vals))
                    if canonical:
                        checks.append(max(float((v-canonical['conditions'][nm]['visual'][k]).abs().max()) for k,v in alt.items()))
                if canonical:alt=canonical['conditions'][nm]['visual']
                z=dict(pair=c['pair'])
                for k in ('scene','question'):
                    own=greedy_score(b,c,f[k]['visual'][-1].double().numpy());other=greedy_score(b,c,alt[k].double().numpy())
                    for m in own:z[k+'_'+m]=own[m]-other[m]
                metrics.append(z)
        assert checks and max(checks)<3e-4
        vals={k:np.array([r[k] for r in metrics]) for k in metrics[0] if k!='pair'}
        for m in ('detail','mrr'):vals['question_minus_scene_'+m]=vals['question_'+m]-vals['scene_'+m]
        out[tag]=dict(max_canonical_difference=max(checks),metrics={k:boot(z,np.ones(len(z)),[r['pair'] for r in metrics]) for k,z in vals.items()})
    dump(ROOT/'results/device_swap.json',out)
if __name__=='__main__':main()
