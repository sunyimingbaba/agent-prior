"""Hold either the pixels or the generated prior fixed and measure the two contributions to the rotation response."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from shared import ROOT,table,rgb,read,dump,sha,setup
from readouts import load_model
from analyze_split import csvout,summary
def main():
    setup();out=ROOT/'val';data=read(out/'rows.json');idx=read(out/'index.json')
    buf=torch.load(out/'features.pt',weights_only=True,map_location='cpu')
    order={(r['image'],r['mode'],r['k']):i for i,r in enumerate(data)}
    t=table('val');m=load_model(False);rows=[];vecs=[]
    with torch.inference_mode():
        for r in read(ROOT/'inputs/retained.json'):
            if r['split']!='val' or r['contrast'] not in ('hv','diag','lr','tb','nose'):continue
            im=rgb(t,r);rot=im.transpose(Image.Transpose.ROTATE_90 if r['contrast'] in ('hv','diag') else Image.Transpose.ROTATE_180)
            images=[m.preprocess(z)[None].cuda() for z in (im,rot)]
            text=buf['text'].double().numpy()
            for k in range(-1,8):
                s='greedy' if k==-1 else f'sample{k}'
                h=[torch.load(ROOT/'samples'/mode/Path(r['image']).stem/(s+'.pt'),map_location='cpu',weights_only=True)[None].cuda() for mode in ('scene','rotated')]
                feat={}
                for a in range(2):
                    for b in range(2):
                        if a==b:v=buf['visual'][order[r['image'],('scene','rotated')[a],k]]
                        else:v=m.img_embeds(images[a],agent_hidden=h[b])['image_feats'][0].cpu()
                        feat[a,b]=v.double().numpy()
                vecs.append(torch.stack([torch.from_numpy(feat[a,b]).float() for a,b in [(0,1),(1,0)]]))
                for w in 'CD':
                    p,q=[idx['texts'].index(s) for s in r[w]]
                    marg=[float(feat[ab]@(text[p]-text[q])) for ab in [(0,0),(0,1),(1,0),(1,1)]]
                    oo,orr,ro,rr=marg
                    pixel=((oo-ro)+(orr-rr))/2;prior=((oo-orr)+(ro-rr))/2
                    assert abs((oo-rr)-(pixel+prior))<1e-12
                    rows.append(dict(image=r['image'],pair=r['pair'],k=k,wording=w,oo=oo,original_rotprior=orr,rot_originalprior=ro,rr=rr,
                                     total=oo-rr,pixel=pixel,prior=prior,interaction=oo-orr-ro+rr))
    csvout(out/'rotation_swap.csv',rows);torch.save(torch.stack(vecs),out/'crossed_vectors.pt')
    per=[]
    for name in sorted({r['image'] for r in rows}):
        g=[r for r in rows if r['image']==name and r['k']>=0];assert len(g)==16
        per.append(dict(image=name,pair=g[0]['pair'],**{key:float(np.mean([r[key] for r in g])) for key in ('total','pixel','prior','interaction')}))
    dump(out/'rotation_swap.json',{k:summary(per,lambda r:r[k]) for k in ['total','pixel','prior','interaction']})
if __name__=='__main__':main()
