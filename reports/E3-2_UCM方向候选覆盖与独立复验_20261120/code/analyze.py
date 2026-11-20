"""Compute raw sign success, candidate coverage and controlled rotation response from archived embeddings."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import itertools
import numpy as np
import torch
from cases import cases,query
from shared import ROOT,read,dump,sha
from metrics import fixed,recalls,ci,csvwrite

def margins(visual,text):
    shape=visual.shape
    s=fixed(visual.reshape(-1,shape[-1]),text).reshape(*shape[:-1],2,2)
    return s[...,0]-s[...,1]

def sign_test(per_source):
    """Exact one-sided conditional randomization, one shared orientation flip per physical source."""
    x=np.array(per_source,dtype=float)
    # Mean signed correctness minus blind 1/2; equal treatment of exact ties.
    null=np.zeros(1)
    for z in x-0.5:
        null=np.concatenate((null+z,null-z))
    observed=(x-0.5).sum()
    return float(np.mean(null>=observed-1e-12))

def endpoint(group,r,w,method,u,v):
    return dict(group=group,image=r['image'],word=w,method=method,raw_original=float(u),raw_rotated=float(v),
                both=int(u>0 and v<0),both_wrong=int(u<0 and v>0),pairacc=float(((u>0)*1.+(v<0)*1.)/2),
                centered=float(u>v)+0.5*float(u==v),orientation_response=float((u-v)/2),bias=float((u+v)/2))

def coverage(group,r,w,method,k,u,v):
    return dict(group=group,image=r['image'],word=w,method=method,k=k,up_positive=int(np.sum(u>0)),rot_positive=int(np.sum(v<0)),
                paired=int(np.sum((u>0)&(v<0))),either_slot_each=int(u.max()>0 and v.min()<0),
                best_upright=float(u.max()),best_rotated=float(-v.min()),weakest_of_bests=float(min(u.max(),-v.min())))

def load_candidates(group,r,out):
    tag=Path(r['image']).stem;f=out/(tag+'.pt');m=read(out/(tag+'.json'))
    assert m['bundle']==sha(f) and m['image']==r['image'] and m['rgb'][0]==r['rgb_sha256']
    assert m['lock']==sha(ROOT/'inputs'/(group+'_lock.json')) and m['queries']==[query(r,c) for c in ('scene','objects')]
    assert m['phrases']==r['A']+r['B'] and m['active']=='2f7e0f3af73b835f2111152bfd76a47f3b811737f24fa51d24c6b6234e005ab6'
    assert m['readout']=='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'
    data=torch.load(f,weights_only=True,map_location='cpu')
    assert len(m['donors'])==(28 if group=='train' else 32)
    assert len(set(x['index'] for x in m['donors']))==len(m['donors'])
    for j,(h,rec) in enumerate(zip(data['own'],m['logs'],strict=True)):
        assert hashlib.sha256(h.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==rec['hidden']
        expected=(('scene','objects')[j//18],j//9%2,j%9)
        assert (rec['cue'],rec['view'],rec['slot'])==expected
        assert rec['seed']==(394000+10*r['row'] if j%9==0 else 1484000+100*r['row']+10*rec['view']+rec['slot']+(0 if group=='train' else 300000))
        assert rec['rounds'] and all(t['ids'] for t in rec['rounds'])
    return m,data

def score_choices(group,r):
    f=ROOT/'choice'/group/(r['image']+'.json');obj=read(f);assert obj['source']==sha(ROOT/'code/choice.py') and obj['design']==sha(ROOT/'inputs/agent_choice_protocol.json')
    assert obj['lock']==sha(ROOT/'inputs'/(group+'_lock.json')) and obj['image']==r['image'] and obj['rgb']==r['rgb_sha256']
    assert len(obj['records'])==12
    index={(a['word'],a['view'],a['order']):a for a in obj['records']};out=[];behavior=[]
    for w in 'AB':
        values=[]
        for v in range(3):
            diff=[]
            for order in (0,1):
                z=index[w,v,order]
                assert z['options']==r[w][::1 if order==0 else -1]
                assert abs(z['valid_mass']-sum(np.exp([z['logprob_a'],z['logprob_b']])))<2e-7
                diff.append((z['logit_a']-z['logit_b'])*(1-2*order))
            values.append(np.mean(diff))
            behavior.append(dict(group=group,image=r['image'],word=w,view=v,order0=diff[0],order1=diff[1],
                     same_order_decision=int(np.sign(diff[0])==np.sign(diff[1])),valid_mass=np.mean([index[w,v,o]['valid_mass'] for o in (0,1)]),
                     top_is_letter=np.mean([index[w,v,o]['top'] in obj['choice_ids'] for o in (0,1)])))
        out.append(endpoint(group,r,w,'agent_choice',values[0],values[1]))
        out.append(endpoint(group,r,w,'agent_gray',values[2],values[2]))
    return out,behavior

def replay_gallery(results):
    folder=ROOT/'gallery';m=read(folder/'replay.json')
    assert sha(folder/'text.pt')==m['text']
    bank=torch.load(folder/'text.pt',weights_only=True,map_location='cpu')
    n=len(m['image_order']);assert n==210 and len(set(m['image_order']))==n
    labels=np.arange(n)[:,None]==np.repeat(np.arange(n),5)[None,:];out={}
    for cue,d in m['parts'].items():
        p=folder/(cue+'.pt');assert sha(p)==d['replayed'] and d['exact']
        mat=fixed(torch.load(p,weights_only=True,map_location='cpu'),bank['native'])[:,bank['caption_order']]
        rr={}
        for prefix,s,l in [('i2t',mat,labels),('t2i',mat.T,labels.T)]:
            for k,v in recalls(s,l).items():rr[prefix+str(k)]=float(v.mean())
        mr=np.mean(list(rr.values()))
        assert abs(mr-m['historical_mR'][cue])<1e-14
        out[cue]=dict(mR=mr,recalls=rr,images=210,captions=1050)
    dump(results/'gallery.json',out)

def main():
    res=ROOT/'results'
    outputs=[];avail=[];allcells=[];foreign=[];choice_behavior=[];traces=[]
    with (ROOT/'inputs/previous_coverage.csv').open(encoding='utf-8-sig') as fp:
        prior={r['image']:r for r in csv.DictReader(fp)}
    strata={'train':('all','old_one_sided','old_two_sided'),'fresh':('all',)}
    def belongs(row,stratum):
        return stratum=='all' or (prior[row['image']]['status']!='both_available')==(stratum=='old_one_sided')
    for group in strata:
        folder=ROOT/'outputs'/group
        proofs=read(folder/'complete.json')
        choiceproof=read(ROOT/'choice'/group/'complete.json')
        assert set(proofs)==set(choiceproof)=={r['image'] for r in cases(group)}
        for r in cases(group):
            tag=Path(r['image']).stem
            assert proofs[r['image']]==sha(folder/(tag+'.json')) and choiceproof[r['image']]==sha(ROOT/'choice'/group/(r['image']+'.json'))
            meta,data=load_candidates(group,r,folder)
            S=margins(data['visual'],data['text'])
            raw=margins(data['clip_visual'],data['clip_text'])
            assert S.shape==(2,36+len(meta['donors']),2)
            rows,behavior=score_choices(group,r);outputs+=rows;choice_behavior+=behavior
            for tr in meta['logs']:
                traces.append(dict(group=group,image=r['image'],cue=tr['cue'],view=tr['view'],slot=tr['slot'],tools=tr['used'],
                              immediate_eos=int(tr['rounds'][0]['ids'] in ([151643],[151645])),tokens=sum(len(x['ids']) for x in tr['rounds']),hidden=tr['hidden']))
            for wi,w in enumerate('AB'):
                u,v=S[:,:,wi];outputs.append(endpoint(group,r,w,'raw_clip',*raw[:,wi]))
                for j,(a,b) in enumerate(zip(u,v,strict=True)):
                    donor=j>=36
                    key=meta['donors'][j-36] if donor else meta['logs'][j]
                    allcells.append(dict(group=group,image=r['image'],word=w,bank='donor' if donor else 'own',index=j,origin_image=key['image'] if donor else r['image'],
                                 origin_view=key['view'],origin_cue='historical scene' if donor else key['cue'],slot=-1 if donor else key['slot'],upright=a,rotated=b))
                for i,cue in enumerate(('scene','objects')):
                    o=i*18;p=o+9
                    outputs.append(endpoint(group,r,w,cue+'_greedy',u[o],v[p]))
                    outputs.append(endpoint(group,r,w,cue+'_mean8',u[o+1:o+9].mean(),v[p+1:p+9].mean()))
                    for k in (1,2,4,8):
                        avail.append(coverage(group,r,w,cue,k,u[o+1:o+1+k],v[p+1:p+1+k]))
                for bank,seg in [('own',slice(0,36)),('donor',slice(36,None)),('combined',slice(None))]:
                    up,rot=u[seg],v[seg]
                    foreign.append(dict(group=group,image=r['image'],word=w,bank=bank,count=len(up),
                              fixed_same_success=int(((up>0)&(rot<0)).any()),fixed_both_count=int(((up>0)&(rot<0)).sum()),
                              independent_success=int(up.max()>0 and rot.min()<0),move_across_zero_upright=int(up.min()<0<up.max()),
                              move_across_zero_rotated=int(rot.min()<0<rot.max()),up_min=up.min(),up_max=up.max(),rot_min=rot.min(),rot_max=rot.max()))
                outputs.append(endpoint(group,r,w,'donor_mean',u[36:].mean(),v[36:].mean()))
    for name,items in [('endpoints',outputs),('coverage',avail),('crossed_margins',allcells),('prior_interventions',foreign),('choice_behavior',choice_behavior),('traces',traces)]:
        csvwrite(res/(name+'.csv'),items)
    table=[];coverage_table=[];bank_table=[];comparisons=[]
    for group,subsets in strata.items():
        for subset in subsets:
            for word in ('A','B','AB'):
                filt=lambda row:row['group']==group and row['word'] in word and belongs(row,subset)
                for method in sorted({x['method'] for x in outputs}):
                    st=[x for x in outputs if x['method']==method and filt(x)]
                    by=defaultdict(list)
                    for x in st:by[x['image']].append(x['centered'])
                    table.append(dict(group=group,subset=subset,word=word,method=method,questions=len(st),sources=len(by),
                              both=sum(x['both'] for x in st),both_wrong=sum(x['both_wrong'] for x in st),pairacc=np.mean([x['pairacc'] for x in st]),
                              centered=np.mean([x['centered'] for x in st]),response=np.mean([x['orientation_response'] for x in st]),
                              response_randomization_p=sign_test([np.mean(v) for v in by.values()])))
                for method in ('scene','objects'):
                    for k in (1,2,4,8):
                        cov=[x for x in avail if x['method']==method and x['k']==k and filt(x)]
                        coverage_table.append(dict(group=group,subset=subset,word=word,method=method,k=k,n=len(cov),accessible=sum(x['either_slot_each'] for x in cov),
                                                  matched=sum(x['paired'] for x in cov),slots=k*len(cov)))
                for bank in ('own','donor','combined'):
                    cells=[x for x in foreign if x['bank']==bank and filt(x)]
                    bank_table.append(dict(group=group,subset=subset,word=word,bank=bank,n=len(cells),fixed=sum(x['fixed_same_success'] for x in cells),
                                      independent=sum(x['independent_success'] for x in cells),cross_upright=sum(x['move_across_zero_upright'] for x in cells),
                                      cross_rotated=sum(x['move_across_zero_rotated'] for x in cells)))
                for left,right in [('objects_mean8','scene_mean8'),('objects_greedy','scene_greedy'),('agent_choice','scene_greedy')]:
                    selected={m:{(x['image'],x['word']):x for x in outputs if x['method']==m and filt(x)} for m in (left,right)}
                    for metric in ('both','pairacc','centered'):
                        diffs=defaultdict(list)
                        for key,a in selected[left].items():diffs[key[0]].append(a[metric]-selected[right][key][metric])
                        comparisons.append(dict(group=group,subset=subset,word=word,left=left,right=right,metric=metric,**ci([np.mean(v) for v in diffs.values()])))
    for name,items in [('deployment_table',table),('coverage_table',coverage_table),('donor_table',bank_table),('paired_contrasts',comparisons)]:
        csvwrite(res/(name+'.csv'),items)
    targeted=next(x for x in coverage_table if (x['group'],x['subset'],x['word'],x['method'],x['k'])==('train','old_one_sided','A','objects',8))
    assert targeted['n']==10
    fresh={x['method']:x for x in table if x['group']=='fresh' and x['word']=='AB'}
    a,b=fresh['objects_mean8'],fresh['scene_mean8']
    dump(res/'decision.json',dict(train_recovered=targeted['accessible'],train_total=10,fresh_targeted_both=a['both'],fresh_scene_both=b['both'],
                                 fresh_questions=a['questions'],majority=targeted['accessible']>5,improves_fresh=a['both']>b['both'],
                                 prioritize_candidate_expansion=bool(targeted['accessible']>5 and a['both']>b['both'])))
    replay_gallery(res)
if __name__=='__main__':main()
