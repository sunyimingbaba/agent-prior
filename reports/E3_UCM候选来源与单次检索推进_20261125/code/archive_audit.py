"""Validate every archived trajectory against the frozen rosters and raw-feature hash chain."""
import hashlib
import io
import json
import sys
import tarfile
import torch
from rosters import ROOT,sha,read,dump

def verify(tag):
    rows=read(ROOT/'inputs'/(tag+'.json'));out={}
    for seed in 'ab':
        folder=ROOT/'records'/tag/seed;cp=read(folder/'raw/complete.json');arc=folder/'trajectories.tar.gz'
        assert sha(arc)==read(folder/'archive.json')['sha256'] and cp['lock']==sha(ROOT/'inputs/lock.json')
        steps={k:0 for k in ('scene','question')};calls={k:0 for k in steps};unique={k:[] for k in steps}
        with tarfile.open(arc) as tf:
            assert json.loads(tf.extractfile('complete.json').read())==cp and len(tf.getnames())==len(rows)+1
            for row in rows:
                nm=row['image'];buf=tf.extractfile(nm+'.pt').read()
                assert hashlib.sha256(buf).hexdigest()==cp['files'][nm]
                obj=torch.load(io.BytesIO(buf),map_location='cpu',weights_only=True)
                feat=torch.load(folder/'features'/(nm+'.pt'),weights_only=True,map_location='cpu')
                assert obj['image']==feat['image']==nm and obj['rgb']==feat['rgb']==row['rgb_sha256']
                assert obj['adapter']==feat['adapter']==cp['adapter'] and feat['source']==cp['files'][nm]
                for m,z in obj['modes'].items():
                    assert z['query']==row['scene_query']+(' '+row['question'] if m=='question' else '')
                    assert z['rngs']==[(10000000 if seed=='a' else 20000000)+row['row']*1000+100*j for j in range(2)]
                    assert len(z['paths'])==16 and all(1<=len(path)<=3 for path in z['paths'])
                    assert z['hidden'].shape==(17,3584) and torch.isfinite(z['hidden']).all()
                    assert all(x==row['feedback'] for x in z['calls']) and obj['feedback']==row['feedback']
                    f=feat['modes'][m];assert f['visual'].shape==(17,768) and len(f['ranks'])==17
                    same={}
                    for i,hr in enumerate(z['hidden']):
                        k=hr.float().numpy().tobytes()
                        if k in same:
                            j=same[k];assert torch.equal(f['visual'][i],f['visual'][j]) and f['ranks'][i]==f['ranks'][j]
                        same[k]=i
                    unique[m].append(len({h.float().numpy().tobytes() for h in z['hidden'][:16]}))
                    steps[m]+=sum(map(len,z['paths']));calls[m]+=sum(len(path)>1 for path in z['paths'])
        out[seed]=dict(images=len(rows),candidate_positions=32*len(rows),sampled_rounds=steps,slots_using_another_round=calls,
            unique_hidden_per_16={k:sum(v)/len(v) for k,v in unique.items()},archive=sha(arc))
    dump(ROOT/'results'/('archive_'+tag+'.json'),out)
if __name__=='__main__':verify(sys.argv[1])
