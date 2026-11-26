import sys,torch
from model_run import ROOT,setup,parent,read,dump,sha,savept,table,rgb,policy,reader,greedy,datasets,TRANS,locked
@torch.inference_mode()
def run(seed,arm):
    lock=locked();setup();torch.set_num_threads(2)
    src=parent(seed) if arm=='parent' else ROOT/'models'/(seed+'_'+arm)
    sig=sha(src/'adapter/adapter_model.safetensors')
    assert sig==(lock['starting'][seed] if arm=='parent' else read(src/'complete.json')['final'])
    m,proc=policy(src);native=reader();data=datasets();held=read(ROOT/'inputs/holdout.json')
    for g,k in [('hold_scene','scene_query'),('hold_first','orig_query')]:
        data[g]=(held,{r['image']:r['feedback'] for r in held},k)
    tables={k:table(k) for k in ('train','val','test')}
    dest=ROOT/'evaluate'/(seed+'_'+arm)
    for tag in (('hold_scene','hold_first') if arm=='parent' else ('hold_scene','hold_first','primary','official')):
        rows,fb,col=data[tag];folder=dest/tag;folder.mkdir(exist_ok=True,parents=True);vs=[]
        for row in rows:
            n=row['image'];f=folder/(n+'.pt');q=row[col]
            if f.exists():
                record=torch.load(f,weights_only=True);assert (record['adapter'],record['query'],record['rgb'])==(sig,q,row['rgb_sha256'])
            else:
                im=rgb(tables[row['split']],row);h,act,calls,_=greedy(m,proc,im,q,fb[n])
                v=native(TRANS(im)[None].cuda(),h.float()[None].cuda())[0].cpu()
                record=dict(image=n,query=q,rgb=row['rgb_sha256'],adapter=sig,actions=act,calls=calls,prior=h,v=v)
                savept(f,record)
            vs.append(record['v'])
        path=dest/(tag+'.pt');savept(path,dict(images=[r['image'] for r in rows],features=torch.stack(vs),adapter=sig))
        dump(folder/'complete.json',dict(features=sha(path),images={r['image']:sha(folder/(r['image']+'.pt')) for r in rows}))
        print(seed,arm,tag,len(vs),flush=True)
if __name__=='__main__':run(*sys.argv[1:])
