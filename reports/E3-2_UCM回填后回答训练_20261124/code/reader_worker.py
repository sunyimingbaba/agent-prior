"""Use the archived FP32 frozen CUP and proxy in their native Python environment."""
import base64,hashlib,json,os,sys
import numpy as np
import torch
from shared import ROOT,REBUILD,read,sha,table,rgb,setup

@torch.inference_mode()
def main(mode):
    output=os.fdopen(os.dup(1),'w',buffering=1)
    os.dup2(2,1)
    setup()
    from readouts import load_model,scorer_for,clip
    reader=load_model(False) if mode!='R1' else None
    is_eval=mode=='eval'
    proxy=None if mode in ('eval','panel') else scorer_for(load_model(True),REBUILD/'relation_candidates_20261120/gallery/train.pt')
    path=REBUILD/'relation_candidates_20261120/native'
    if mode in ('R2','panel'):
        index=read(path/'index.json')
        assert sha(path/'features.pt')==index['sha256']
        bank=torch.load(path/'features.pt',map_location='cpu',weights_only=True)
        features=bank['features'].double()
        even=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['tokens'].tolist()])
    t=table('val' if is_eval else 'train')
    src=REBUILD/('paired_views_20261120/inputs/validation.json' if is_eval else 'matched_policy_20261120/inputs/train.json')
    cases={z['image']:z for z in read(src)}
    if is_eval:
        gallery=torch.load(REBUILD/'paired_views_20261120/inputs/retrieval_text.pt',map_location='cpu',weights_only=True)
        raw=clip.tokenize([q for qs in t['caption'].to_pylist() for q in qs],context_length=77-reader.prompt_length)
        assert torch.equal(raw,gallery['native_tokens'][gallery['caption_order']])
        assert (reader.txt_embeds(raw[:4])['text_feats']-gallery['native'][gallery['caption_order'][:4]]).abs().max()<2e-5
    print('READY',file=output)
    for line in sys.stdin:
        q=json.loads(line);c=cases[q['image']]
        h=torch.from_numpy(np.frombuffer(base64.b64decode(q['hidden']),dtype=np.float32).copy().reshape(-1,3584))
        pix=reader.preprocess(rgb(t,c))[None] if reader else None
        result=[];cache={}
        for z in h:
            key=z.numpy().tobytes()
            if key not in cache:
                row={}
                if proxy:row['rank']=proxy.score(z.tolist(),c['row'])[0]
                if reader:
                    vis=reader.img_embeds(pix,agent_hidden=z[None])['image_feats'][0]
                    if is_eval:row['feature']=vis.tolist()
                    else:
                        mapping=index['maps'][c['image']]
                        assert mapping['captions']==t['caption'][c['row']].as_py()
                        allowed=even.copy();allowed[mapping['positive']]=False
                        sims=features@vis.double()
                        row['margin']=float(sims[mapping['first']]-sims[allowed].max())
                        if mode=='panel':
                            other=~even
                            other[mapping['positive']]=False
                            alt=sorted(set(mapping['positive'])-{mapping['first']})
                            assert alt
                            best=sims[alt].max()
                            higher=int((sims[other]>best).sum())
                            ties=int((sims[other]==best).sum())
                            row['mrr']=float(np.mean(1/np.arange(higher+1,higher+ties+2)))
                cache[key]=row
            result.append(cache[key])
        print(json.dumps(result,allow_nan=False),file=output)
if __name__=='__main__':main(sys.argv[1])
