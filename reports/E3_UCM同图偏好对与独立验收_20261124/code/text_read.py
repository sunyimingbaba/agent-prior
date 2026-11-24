import hashlib,sys,time,numpy as np,torch
from env_pair import ROOT,OLD,read,dump,rows,table,rgb,setup,sha
from readouts import load_model
@torch.inference_mode()
def main():
    setup();m=load_model(False);t=table();rs=rows();banks={}
    for sub in ('','confirm'):
        bank=torch.load(OLD/sub/'records/text.pt',weights_only=True)
        even=np.array([int(hashlib.sha256(','.join(map(str,qs)).encode()).hexdigest()[-1],16)%2==0 for qs in bank['native_tokens'].tolist()])
        banks[sub]=(bank,even)
    for seed in 'ab':
        src=ROOT/'records'/(seed+'_text_states.pt')
        while not src.is_file():time.sleep(1)
        dat=torch.load(src,weights_only=True);out=[]
        for r in read(ROOT/'inputs/candidates.json')[seed]:
            nm=r['image'];c=rs[nm];v=dat['images'][nm];assert c['rgb_sha256']==v['rgb'];bank,even=banks[r['source']];ix=bank['maps'][nm]
            px=m.preprocess(rgb(t,c))[None];fs=[m.img_embeds(px,agent_hidden=z.float()[None])['image_feats'][0].double() for z in v['hidden']]
            h=bank['hard'].double();ne=even.copy();np0=ix['native'];ne[np0]=False;nn=bank['native'].double()
            sc=[]
            for f in fs:
                vals=nn@f;hh=h@f;sc.append(dict(N=float(vals[np0[0]]-vals[ne].max()),**{k:float(hh[ix[k][0]]-hh[ix[k][1]]) for k in 'AB'}))
            out.append(dict(image=nm,attribute=c['attribute'],scene=c['scene_id'],side=c['side'],pair=c['pair'],scores=sc))
        dump(ROOT/'records'/(seed+'_text_read.json'),dict(states=sha(src),images=out))
if __name__=='__main__':main()
