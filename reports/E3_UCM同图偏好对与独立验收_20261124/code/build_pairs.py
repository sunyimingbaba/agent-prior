"""Recover second-turn alternatives and rank their frozen downstream effects within source image."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];REP=ROOT.parent;HIST=REP/'E3-2_UCM有效图组定向短训_20261124'
sys.path.insert(0,str(HIST/'code'))
from verify_screen import unpack
from score import OLD,read,load,sha,dump
def main():
    torch.set_num_threads(2);allrows={};summary={}
    for seed in ('a','b'):
        out=[];paired=0;post=0
        for sub in ('','confirm'):
            base=OLD/sub;bank=load(base/'records/text.pt');ext=HIST/'screen/records' if sub else base/'records'
            first=unpack(base/'records',seed);second=unpack(ext,'extra_'+seed);h=bank['hard'].double().numpy();native=bank['native'].double().numpy()
            even=np.array([int(hashlib.sha256(','.join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in bank['native_tokens'].tolist()])
            for c in read(base/'inputs/panel.json'):
                nm=c['image'];s,_=first[nm]
                if not s['post_tool']:continue
                post+=1;m=bank['maps'][nm];seen={};neg=even.copy();neg[m['native']]=False
                for j,rec in enumerate([first[nm],second[nm]]):
                    acts,vec=rec;v=vec['visual'].double().numpy()
                    abr=np.stack([v@(h[m[l][0]]-h[m[l][1]]) for l in 'AB'],1)
                    sim=v@native.T;nat=sim[:,m['native'][0]]-sim[:,neg].max(1)
                    for i,ids in enumerate(acts['second']):
                        k=tuple(ids);data=dict(ids=ids,third=acts['third'][i],A=float(abr[i,0]),B=float(abr[i,1]),N=float(nat[i]),draws=[0,0])
                        if k not in seen:seen[k]=data
                        z=seen[k];assert z['third']==data['third'] and max(abs(z[l]-data[l]) for l in ('A','B','N'))<1e-10
                        z['draws'][j]+=1
                alts=list(seen.values());pa=[]
                for i,u in enumerate(alts):
                    if min(u['A'],u['B'])<=0:continue
                    for j,v in enumerate(alts):
                        if max(v['A'],v['B'])>=0:continue
                        pa.append((i,j))
                if pa:paired+=1
                out.append(dict(image=nm,source=sub,row=c['row'],rgb=c['rgb_sha256'],first=s['first'],tool=s['tool_calls'][0],attribute=c['attribute'],scene=c['scene_id'],side=c['side'],alternatives=alts,pairs=pa))
        allrows[seed]=out
        raw=[(r,u,v) for r in out for i,j in r['pairs'] for u,v in [(r['alternatives'][i],r['alternatives'][j])]]
        summary[seed]=dict(post=post,images=paired,distinct_pairs=len(raw),concordant=sum(u['N']>=v['N'] for r,u,v in raw),distinct_details=len({r['attribute'] for r,u,v in raw}),answer_lengths=sorted(set(len(w['ids']) for r,u,v in raw for w in (u,v))),third_positives=sum(bool(u['third']) for r,u,v in raw),third_negatives=sum(bool(v['third']) for r,u,v in raw))
    dump(ROOT/'inputs/candidates.json',allrows);dump(ROOT/'results/pair_inventory.json',summary)
if __name__=='__main__':main()
