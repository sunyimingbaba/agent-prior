"""Verify all returned generated priors, vectors, source images and matched final action trajectories."""
import csv,tarfile,io,hashlib,json
import numpy as np,torch
from report_io import ROOT,LAST,FORM,load,read,sha,dump
def main():
    index=read(ROOT/'archives/index.json');panels={sub:{r['image']:r for r in read(p)} for sub,p in [
        ('primary',ROOT/'inputs/panel.json'),('official',FORM/'inputs/ucm.json'),('prior148',LAST/'confirm/inputs/panel.json'),('prior142',LAST/'reward_confirm/inputs/panel.json')]}
    with (ROOT/'results/per_image.csv').open() as f:outcomes={(r['arm'],r['seed'],r['subset'],r['image']):float(r['CDE']) for r in csv.DictReader(f) if r['CDE']}
    allstats={};history={};identity={};num=0
    for filename,h in sorted(index['files'].items()):
        tag=filename[:-7];seed,arm=tag.split('_')
        ar=ROOT/'archives'/filename;assert sha(ar)==h;traces={};proofs={};stats={}
        with tarfile.open(ar,'r|gz') as tf:
            for mem in tf:
                if not mem.isfile():continue
                tg,sub,file=mem.name.split('/');assert tg==tag
                raw=tf.extractfile(mem).read()
                if file=='complete.json':proofs[sub]=json.loads(raw);continue
                nm=file[:-3];assert file.endswith('.pt')
                z=torch.load(io.BytesIO(raw),weights_only=True);c=panels[sub][nm]
                assert z['image']==nm and z['rgb']==c['rgb_sha256'] and z['query']==c['orig_query' if sub=='official' else 'scene_query']
                v=z['prior'];assert v.shape==(3584,) and v.dtype==torch.bfloat16 and torch.isfinite(v).all()
                assert len(z['rounds'])>0 and len(z['rounds'])<=3 and len(z['tools'])<=len(z['rounds'])
                traces.setdefault(sub,{})[nm]=(hashlib.sha256(raw).hexdigest(),z['adapter'])
                stats.setdefault(sub,[]).append(len(z['tools']))
                if arm in ('R2','R3') and sub!='official':
                    seq=tuple(tuple(r) for r in z['rounds']);tools=tuple(z['tools'])
                    key=seed,sub,nm
                    if arm=='R2':history[key]=(seq,tools,v)
                    else:
                        a,b,w=history.pop(key);same=seq==a and tools==b
                        k=seed+'_'+sub;arr=identity.setdefault(k,[])
                        arr.append([same,tools==b,float(torch.nn.functional.cosine_similarity(v[None].float(),w[None].float())),outcomes['R3',seed,sub,nm]-outcomes['R2',seed,sub,nm]])
        for sub,files in traces.items():
            vec=load(ROOT/'vectors'/(tag+'_'+sub+'.pt'))
            assert proofs[sub]['adapter']==vec['adapter'] and set(panels[sub])==set(files)==set(vec['images'])
            assert dict(zip(vec['images'],vec['sources']))=={n:p[0] for n,p in files.items()}==proofs[sub]['files']
            assert all(sig==vec['adapter'] for _,sig in files.values())
            num+=len(files)
        allstats[tag]={sub:dict(images=len(v),with_tools=sum(x>0 for x in v),calls=sum(v)) for sub,v in stats.items()}
    assert not history
    behavior={}
    for key,vs in identity.items():
        x=np.array(vs);mask=x[:,0]>0
        behavior[key]=dict(images=len(x),same_all_actions=int(mask.sum()),same_tool_returns=int(x[:,1].sum()),mean_prior_cos=float(x[:,2].mean()),net_correct_votes_same_actions=round(3*x[mask,3].sum()),net_correct_votes_changed_actions=round(3*x[~mask,3].sum()))
    dump(ROOT/'results/archive_verification.json',dict(traces=num,archives=len(index['files']),vector_files=len(index['vectors']),tool_usage=allstats,R3_vs_R2=behavior))
if __name__=='__main__':main()
