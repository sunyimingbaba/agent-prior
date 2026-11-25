"""Verify all sampled first-caption groups and the matched scene-run sampling conditions."""
from hashlib import sha256
from io import BytesIO
import json
import tarfile
import torch
from compare import OUT, read, dump, sha, load

def groups(path):
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            yield member.name,archive.extractfile(member).read()

def control_rows(seed):
    scene = {}
    root = OUT/'archives'
    for tag in ('fresh','extension','extra'):
        prefix = 'covered' if tag=='extra' else 'candidate'
        for name,blob in groups(root/(prefix+'_'+tag+'_'+seed+'.tar.gz')):
            if name=='complete.json': continue
            z=torch.load(BytesIO(blob),weights_only=True)
            scene[z['image']]=(z['rgb'],z['feedback'],z['adapter'],z['rngs'],z['query'])
    for lock in OUT.parent.glob('E3_UCM中性观察问题*/inputs/lock.json'):
        for j,c in enumerate(read(lock)['rows']):
            p=lock.parents[1]/'records'/seed/(c['image']+'.json')
            z=read(p)
            base=(10000000 if seed=='a' else 20000000)+1000*j
            scene[c['image']]=(z['rgb'],z['feedback'],z['adapter'],[base,base+100],z['groups']['scene']['query'])
    return scene

def main():
    rows = {r['image']:r for r in read(OUT/'inputs/real_roster.json')}
    index = read(OUT/'archives/real_index.json')
    report={}
    for seed in 'ab':
        expected=control_rows(seed)
        file='real_all_'+seed+'.tar.gz'
        path=OUT/'archives'/file
        assert sha(path)==index[file]['sha']
        completed=set()
        for name,blob in groups(path):
            if name=='complete.json':
                cp=json.loads(blob)
                assert sha256(blob).hexdigest()==index[file]['manifest'] and set(cp['files'])==set(rows)
                continue
            z=torch.load(BytesIO(blob),weights_only=True)
            n=z['image'];r=rows[n];assert name==n+'.pt' and n not in completed
            assert z['adapter']==index[file]['adapter']==cp['adapter']
            assert z['rgb']==r['rgb_sha256'] and z['query']==r['orig_query'] and z['feedback']==r['feedback']
            assert (z['rgb'],z['feedback'],z['adapter'],z['rngs'],r['scene_query'])==expected[n]
            assert z['rngs']==[r['rng_'+seed],r['rng_'+seed]+100]
            assert z['hidden'].shape==(17,3584) and torch.isfinite(z['hidden']).all() and len(z['paths'])==16
            v=load(OUT/'records/real'/seed/'features'/name)
            assert v['source']==sha256(blob).hexdigest()==cp['files'][n] and v['rgb']==z['rgb'] and v['adapter']==z['adapter']
            assert v['visual'].shape==(17,768) and torch.isfinite(v['visual']).all()
            completed.add(n)
        assert completed==set(rows)
        report[seed]=dict(groups=len(completed),sampled=16*len(completed),greedy=len(completed),adapter=cp['adapter'],archive=sha(path))
    dump(OUT/'results/real_integrity.json',report)

if __name__=='__main__':main()
