import sys,time,tarfile,torch
from runtime import ROOT,FORM,sha,read,dump,savept,table,rgb,setup
from eval_io import TAGS,weights,datasets,subsets
from readouts import load_model,clip
@torch.inference_mode()
def main():
    setup();model=load_model(False);ds=datasets();tables={s:table(s) for s in ('train','val','test')}
    text=sorted({q for sub,(rows,_,_) in ds.items() if sub!='official' for c in rows for k in 'CDE' for q in c[k]})
    tt=clip.tokenize(text,context_length=77-model.prompt_length)
    assert len(torch.unique(tt,dim=0))==len(text)
    hard=torch.cat([model.txt_embeds(b)['text_feats'] for b in tt.split(12)])
    ix={s:j for j,s in enumerate(text)}
    maps={sub:{c['image']:{k:[ix[q] for q in c[k]] for k in 'CDE'} for c in rows} for sub,(rows,_,_) in ds.items() if sub!='official'}
    savept(ROOT/'records/text.pt',dict(text=text,hard=hard,tokens=tt,maps=maps,panel=sha(ROOT/'inputs/panel.json')))
    gallery=FORM/'results/ucm_text.pt';archives={}
    for tag in TAGS:
        _,sig=weights(tag)
        archive=ROOT/'archives'/(tag+'.tar.gz');part=archive.with_suffix('.part')
        with tarfile.open(part,'w:gz',compresslevel=3) as tar:
            for sub in subsets(tag):
                rows,answers,query=ds[sub];names=[];sources=[];vectors=[]
                folder=ROOT/'greedy'/tag/sub
                for c in rows:
                    nm=c['image'];p=folder/(nm+'.pt')
                    while not p.is_file():time.sleep(1)
                    state=torch.load(p,weights_only=True);assert state['rgb']==c['rgb_sha256'] and state['image']==nm and state['adapter']==sig and state['query']==c[query] and all(q==answers[nm] for q in state['tools'])
                    v=model.img_embeds(model.preprocess(rgb(tables[c['split']],c))[None],agent_hidden=state['prior'].float()[None])['image_feats'][0]
                    assert abs(v.norm().item()-1)<1e-4 and torch.isfinite(v).all()
                    names.append(nm);sources.append(sha(p));vectors.append(v)
                while not (folder/'complete.json').exists():time.sleep(1)
                assert read(folder/'complete.json')['files']==dict(zip(names,sources))
                savept(ROOT/'vectors'/(tag+'_'+sub+'.pt'),dict(images=names,features=torch.stack(vectors),sources=sources,adapter=sig))
                tar.add(folder,arcname=tag+'/'+sub)
        part.replace(archive);archives[archive.name]=sha(archive);print('archived',tag,flush=True)
    dump(ROOT/'archives/index.json',dict(files=archives,text=sha(ROOT/'records/text.pt'),gallery=sha(gallery),vectors={f.name:sha(f) for f in sorted((ROOT/'vectors').glob('*.pt'))}))
if __name__=='__main__':main()
