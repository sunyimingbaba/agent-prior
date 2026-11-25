"""Form a capped, hash-ordered roster using only the other seed's scene flag."""
import hashlib
from collections import defaultdict
import torch
from compare import OUT,read,load,sha,dump
from single_scores import FORM,RFPO

def main():
    rows=read(OUT/'inputs/real_roster.json')
    original=read(OUT/'records/mining_images.json')
    flags={(z['image'],z['to_seed']):z['source']['changed'] for z in original}
    assert read(OUT/'results/mining.json')['eligible']
    forbidden={r['rgb_sha256'] for p in (RFPO/'inputs/panel.json',FORM/'inputs/ucm.json') for r in read(p)}
    plans={};keys=set()
    for s in 'ab':
        key=lambda r:hashlib.sha256(('e3-mined-20261125-v1|'+s+'|'+r['rgb_sha256']).encode()).hexdigest()
        for group in ('selected','uniform'):
            out=[];pairs=set()
            for r in sorted(rows,key=key):
                if group=='selected' and not flags[r['image'],s]:continue
                pair=(r['tag'],r['pair'])
                if pair in pairs:continue
                assert r['rgb_sha256'] not in forbidden and r['split']=='train'
                out.append({k:r[k] for k in ('image','rgb_sha256','pair','row','tag','orig_query','feedback','split')})
                pairs.add(pair);keys.add(r['image'])
                if len(out)==24:break
            assert len(out)==24;plans[s+'_'+group]=out
    q=OUT/'inputs/mined_roster.json';dump(q,plans)
    a,b=[load(p) for p in [OUT/'records/text.pt',OUT/'records/covered/bank/text.pt']]
    assert torch.equal(a['native'],b['native']) and torch.equal(a['native_tokens'],b['native_tokens'])
    hard=[];maps={}
    for r in rows:
        n=r['image']
        if n not in keys:continue
        z=a if r['tag'] in ('fresh','extension') else b;ix=z['maps'][n];mp=dict(native=ix['native'])
        for k in 'AB':
            mp[k]=[len(hard),len(hard)+1];hard.extend(z['hard'][ix[k]].unbind(0))
        maps[n]=mp
    p=OUT/'inputs/mined_training_bank.pt'
    torch.save(dict(native=a['native'],native_tokens=a['native_tokens'],hard=torch.stack(hard),maps=maps),p)
    dump(OUT/'inputs/mined_lock.json',dict(protocol=sha(OUT/'筛图后从现有R2继续训练.md'),training=sha(q),bank=sha(p),evidence=sha(OUT/'results/mining.json'),
        implementation={k:sha(OUT/'code'/k) for k in ('mined_train.py','mined_worker.py','exact_reward.py','mined_eval.py')}))
if __name__=='__main__':main()
