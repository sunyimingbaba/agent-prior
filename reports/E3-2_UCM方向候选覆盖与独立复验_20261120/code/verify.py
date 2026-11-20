"""Independent score arithmetic and provenance checks against the previous baseline, pixel bytes and output tables."""
import argparse,csv,hashlib,io
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from cases import cases
from shared import ROOT,read,sha,dump

def records(path):
    with path.open(encoding='utf-8-sig') as fp:return list(csv.DictReader(fp))

def main(prior,arrow):
    res=ROOT/'results'
    logged={(r['group'],r['image'],r['word'],int(r['index'])):r for r in records(res/'crossed_margins.csv')}
    coverage={(r['group'],r['image'],r['word'],r['method'],int(r['k'])):r for r in records(res/'coverage.csv')}
    endpoint={(r['group'],r['image'],r['word'],r['method']):r for r in records(res/'endpoints.csv')}
    parent=prior/'eval/base_00'
    archive=torch.load(parent/'trainpanel.pt',weights_only=True,map_location='cpu');info=read(parent/'trainpanel.json')
    assert sha(parent/'trainpanel.pt')==info['bundle']
    maxdiff=0.;cells=0;checked_cov=0;replayed=0;pixels=0;forced=0;second=0
    for group in ('train','fresh'):
        t=pa.ipc.open_file(str(arrow/('ucm_caption_karpathy_label_'+('train' if group=='train' else 'val')+'.arrow'))).read_all()
        for row in cases(group):
            where=ROOT/'outputs'/group/Path(row['image']).stem
            d=torch.load(where.with_suffix('.pt'),weights_only=True,map_location='cpu');meta=read(where.with_suffix('.json'))
            im=Image.open(io.BytesIO(t['image'][row['row']].as_py())).convert('RGB')
            assert [hashlib.sha256(x.tobytes()).hexdigest() for x in (im,im.transpose(Image.Transpose.ROTATE_180))]==meta['rgb']
            pixels+=2
            for donor in meta['donors']:
                old=info['records'][donor['index']]
                assert old['image']==donor['image'] and old['view']==donor['view'] and old['hidden_sha']==donor['hidden']
                if group=='train':assert cases('train')[donor['index']//2]['pair']!=row['pair']
            if group=='train':
                for j in (0,1):
                    k=next(i for i,z in enumerate(info['records']) if z['image']==row['image'] and z['view']==('rotated' if j else 'upright'))
                    assert torch.equal(d['own'][9*j],archive['hidden'][k]) and torch.equal(d['visual'][j,j*9],archive['visual'][k])
                    assert meta['logs'][9*j]['rounds']==info['records'][k]['rounds'];replayed+=1
            x=d['visual'].double().numpy();delta=d['text'].double().numpy().reshape(2,2,-1)
            s=np.einsum('vid,wd->viw',x,delta[:,0]-delta[:,1],optimize=False)
            for i,w in enumerate('AB'):
                for j in range(x.shape[1]):
                    z=logged[group,row['image'],w,j]
                    err=max(abs(s[0,j,i]-float(z['upright'])),abs(s[1,j,i]-float(z['rotated'])))
                    maxdiff=max(maxdiff,err);cells+=2
                def check(u,v,label):
                    nonlocal second
                    z=endpoint[group,row['image'],w,label]
                    assert (int(z['both']),float(z['pairacc']),float(z['centered']))==(int(u>0 and v<0),(int(u>0)+int(v<0))/2,float(u>v)+0.5*float(u==v))
                    assert max(abs(float(z['raw_original'])-u),abs(float(z['raw_rotated'])-v))<1e-11
                    second+=1
                for cue,offset in (('scene',0),('objects',18)):
                    su=s[0,offset:offset+9,i];sr=s[1,offset+9:offset+18,i]
                    check(su[0],sr[0],cue+'_greedy');check(su[1:].mean(),sr[1:].mean(),cue+'_mean8')
                    for k in (1,2,4,8):
                        z=coverage[group,row['image'],w,cue,k]
                        u=[n>0 for n in su[1:k+1]];v=[n<0 for n in sr[1:k+1]]
                        assert (int(z['either_slot_each']),int(z['paired']))==(int(any(u) and any(v)),sum(a and b for a,b in zip(u,v)));checked_cov+=1
                check(s[0,36:,i].mean(),s[1,36:,i].mean(),'donor_mean')
                cv=d['clip_visual'].double().numpy()@(d['clip_text'][2*i].double().numpy()-d['clip_text'][2*i+1].double().numpy())
                check(*cv,'raw_clip')
                choices=read(ROOT/'choice'/group/(row['image']+'.json'))['records'];difference=[]
                for view in range(3):
                    opts=sorted((q for q in choices if q['word']==w and q['view']==view),key=lambda z:z['order'])
                    assert opts[0]['options']==opts[1]['options'][::-1]==row[w]
                    gap=((opts[0]['logit_a']-opts[0]['logit_b'])+(opts[1]['logit_b']-opts[1]['logit_a']))/2
                    difference.append(gap);forced+=2
                check(difference[0],difference[1],'agent_choice');check(difference[2],difference[2],'agent_gray')
    assert cells==2*len(logged) and checked_cov==len(coverage) and second==len(endpoint) and maxdiff<1e-12
    p=ROOT/'gallery';receipt=read(p/'replay.json')
    for cue in ('native','generic'):
        restored=torch.load(p/(cue+'.pt'),weights_only=True,map_location='cpu')
        orig=torch.load(parent/(cue+'.pt'),weights_only=True,map_location='cpu')
        assert torch.equal(restored,orig['visual']) and sha(parent/(cue+'.pt'))==receipt['parts'][cue]['input']
    dump(res/'verification.json',dict(raw_margins=cells,max_alternative_absolute_error=maxdiff,nested_coverage_rows=checked_cov,method_source_word_rows=second,
         forced_option_prompts=forced,pixel_hashes=pixels,archived_greedy_train_replays=replayed,historical_gallery_embeddings=420,archived_priors_sha=info['bundle']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prior',type=Path);p.add_argument('arrow',type=Path);a=p.parse_args();main(a.prior,a.arrow)
