import sys,torch
from model_run import ROOT,setup,locked,parent,table,rgb,reader,policy,TRANS,read,dump,sha,savept,greedy
@torch.inference_mode()
def main(seed,arm):
    lock=locked();setup();torch.set_num_threads(2)
    src=parent(seed) if arm=='parent' else ROOT/'models'/(seed+'_'+arm)
    sig=sha(src/'adapter/adapter_model.safetensors')
    assert sig==(lock['starting'][seed] if arm=='parent' else read(src/'complete.json')['final'])
    out=ROOT/'fitcheck';out.mkdir(exist_ok=True);f=out/(seed+'_'+arm+'.pt');proof=f.with_suffix('.json')
    if proof.exists():assert sha(f)==read(proof)['features'];return
    model,proc=policy(src);native=reader();tb=table();rows=[]
    for r in read(ROOT/'inputs/train.json')[seed]:
        im=rgb(tb,r);pix=TRANS(im)[None].cuda()
        for view in ('scene','first'):
            q=r['scene_query' if view=='scene' else 'orig_query']
            h,action,calls,_=greedy(model,proc,im,q,r['feedback'])
            v=native(pix,h.float()[None].cuda())[0].cpu()
            rows.append(dict(image=r['image'],view=view,query=q,rgb=r['rgb_sha256'],actions=action,calls=calls,prior=h,visual=v))
    savept(f,dict(adapter=sig,rows=rows));dump(proof,dict(features=sha(f),rows=len(rows),adapter=sig))
if __name__=='__main__':main(*sys.argv[1:])
