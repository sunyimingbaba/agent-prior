import sys,torch
from runtime import ROOT,read,dump,sha,savept,setup,table,rgb
from eval_io import datasets,weights,subsets,TAGS
from policy import policy
from prefixes import greedy

@torch.inference_mode()
def run(tag):
    assert tag in TAGS;setup();ds=datasets();path,sig=weights(tag);model,proc=policy(path)
    tables={s:table(s) for s in ('train','val','test')}
    for sub in subsets(tag):
        rows,answers,query=ds[sub]
        dest=ROOT/'greedy'/tag/sub;dest.mkdir(parents=True,exist_ok=True)
        for c in rows:
            nm=c['image'];p=dest/(nm+'.pt')
            if p.is_file():
                old=torch.load(p,weights_only=True);assert old['adapter']==sig and old['rgb']==c['rgb_sha256'] and old['query']==c[query];continue
            h,actions,calls,_=greedy(model,proc,rgb(tables[c['split']],c),c[query],answers[nm])
            savept(p,dict(image=nm,rgb=c['rgb_sha256'],adapter=sig,query=c[query],prior=h,rounds=actions,tools=calls))
        dump(dest/'complete.json',dict(adapter=sig,files={c['image']:sha(dest/(c['image']+'.pt')) for c in rows}))
        print(tag,sub,len(rows),flush=True)
if __name__=='__main__':run(sys.argv[1])
