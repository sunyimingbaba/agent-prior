import csv,sys
import numpy as np
from report_io import ROOT,LAST,HIST,FORM,OLD,load,read,dump,sha,trainrows
sys.path.insert(0,str(HIST/'code'))
from score import boot,ranked_hits
sys.path.insert(0,str(FORM/'code'))
from metrics import fixed,recalls

def main():
    lock=read(ROOT/'inputs/panel_lock.json');index=read(ROOT/'archives/index.json')
    assert sha(ROOT/'inputs/panel.json')==lock['panel'] and sha(ROOT/'records/text.pt')==index['text']
    assert sha(FORM/'results/ucm_text.pt')==index['gallery']
    bank=load(ROOT/'records/text.pt');gallery=load(FORM/'results/ucm_text.pt')
    panels={sub:read(p) for sub,p in [('primary',ROOT/'inputs/panel.json'),('prior148',LAST/'confirm/inputs/panel.json'),('prior142',LAST/'reward_confirm/inputs/panel.json'),('official',FORM/'inputs/ucm.json')]}
    assert len({c['rgb_sha256'] for s,cc in panels.items() if s!='official' for c in cc})==sum(len(cc) for s,cc in panels.items() if s!='official')
    train=trainrows()
    seen={(train[n]['scene_id'],train[n]['attribute']) for n in read(ROOT/'inputs/roster.json')['order']}
    levels={};items={};recalls_checked=0;raw=[];replay={}
    for seed in 'ab':
        for arm in ('R0','deployed','R2','R3'):
            tag=seed+'_'+arm;levels[tag]={}
            for sub,rows in panels.items():
                if arm=='R0' and sub.startswith('prior'):continue
                f=ROOT/'vectors'/(tag+'_'+sub+'.pt');assert sha(f)==index['vectors'][f.name]
                v=load(f);names=[r['image'] for r in rows];assert v['images']==names
                if arm=='deployed' and sub!='primary':
                    saved=LAST/('pilot/vectors/'+seed+'_start_reward_anchor.pt' if sub=='prior142' else 'confirm/'+seed+'_start_'+('official' if sub=='official' else 'blind')+'.pt')
                    check=load(saved);assert check['images']==names and check['adapter']==v['adapter']
                    num=float((check['features']-v['features']).abs().max());replay[tag+'_'+sub]=num;assert num<1e-5
                if sub=='official':
                    n=len(rows);mask=np.arange(n)[:,None]==np.repeat(np.arange(n),5)[None,:]
                    s=fixed(v['features'],gallery['native'])[:,gallery['caption_order']]
                    ir=recalls(s,mask);tr=recalls(s.T,mask.T)
                    a=np.stack(list(ir.values()),1);b=np.stack(list(tr.values()),1).reshape(n,5,3).mean(1)
                    matrix=(v['features'].double()@gallery['native'].double().T)[:,gallery['caption_order']].numpy()
                    assert np.allclose(a,ranked_hits(matrix,mask),atol=1e-10) and np.allclose(b,ranked_hits(matrix.T,mask.T).reshape(n,5,3).mean(1),atol=1e-10)
                    recalls_checked+=6;hits=np.column_stack((a,b));vals=dict(mR=hits.mean(1),I2T=a.mean(1),T2I=b.mean(1))
                    levels[tag][sub]={'recalls':hits.mean(0).tolist()}
                else:
                    s=fixed(v['features'],bank['hard']);vals={};idx=np.arange(len(rows))
                    for k in 'CDE':
                        cols=np.array([bank['maps'][sub][n][k] for n in names]);vals[k]=(s[idx,cols[:,0]]>s[idx,cols[:,1]]).astype(float)
                    vals['CDE']=np.stack(list(vals.values())).mean(0);levels[tag][sub]={}
                levels[tag][sub].update({k:float(v.mean()) for k,v in vals.items()})
                items[(seed,arm,sub)]=vals
                for j,c in enumerate(rows):raw.append(dict(seed=seed,arm=arm,subset=sub,image=c['image'],pair=c.get('pair',''),**{k:float(a[j]) for k,a in vals.items()}))
    differences={}
    for left,right in [('R3','R2'),('R3','deployed'),('R2','deployed'),('R3','R0'),('R2','R0')]:
        outputs={};allold=[]
        for sub,rows in panels.items():
            if ('a',right,sub) not in items:continue
            labels=[r.get('pair',r['image']) for r in rows];out={}
            for k in items[('a',left,sub)]:
                diff=np.array([items[(s,left,sub)][k]-items[(s,right,sub)][k] for s in 'ab'])
                assert diff.shape==(2,len(rows))
                out[k]=dict(**boot(diff.mean(0),labels),seeds=diff.mean(1).tolist())
                if k=='CDE':
                    if sub.startswith('prior'):allold.append((diff,[sub+'_'+i for i in labels]))
                    out[k]['gained_lost']={s:[sum(int((items[(s,left,sub)][q]-items[(s,right,sub)][q]>0).sum()) for q in 'CDE'),sum(int((items[(s,left,sub)][q]-items[(s,right,sub)][q]<0).sum()) for q in 'CDE')] for s in 'ab'}
                    if sub=='primary':
                        out[k]['by_scene_resample']=boot(diff.mean(0),[r['scene_id'] for r in rows])
                        groups={'val':[r['split']=='val' for r in rows],'test':[r['split']=='test' for r in rows],'seen':[(r['scene_id'],r['attribute']) in seen for r in rows]}
                        groups['unseen']=[not b for b in groups['seen']]
                        out[k]['strata']={key:dict(images=sum(mask),mean=float(diff[:,mask].mean()),seeds=diff[:,mask].mean(1).tolist()) for key,mask in groups.items() if any(mask)}
            outputs[sub]=out
        if len(allold)==2:
            z=np.concatenate([a for a,g in allold],axis=1);labels=sum((g for a,g in allold),[])
            outputs['prior290']=dict(CDE=dict(**boot(z.mean(0),labels),seeds=z.mean(1).tolist()))
        differences[left+'-'+right]=outputs
    dump(ROOT/'results/outcomes.json',dict(levels=levels,comparisons=differences,recall_checks=recalls_checked,retained_baseline_replay=replay))
    with (ROOT/'results/per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed','arm','subset','image','pair','C','D','E','CDE','mR','I2T','T2I']);w.writeheader();w.writerows(raw)
if __name__=='__main__':main()
