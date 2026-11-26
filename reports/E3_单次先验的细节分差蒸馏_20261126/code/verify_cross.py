from pathlib import Path
import io,hashlib,json,tarfile,sys
import torch
from local import ROOT,FORM,load,read,dump,sha
def main(mode="base"):
    stage=ROOT if mode=="base" else ROOT/mode
    idx=read(stage/'archives/extra_index.json');assert sha(stage/'archives/all_execution_logs.tar.gz')==idx['logs']
    rows=read(FORM/'inputs/rsitmd.json');by={r['image']:r for r in rows};out={}
    for key,h in idx['cross'].items():
        p=stage/'archives'/('cross_'+key+'.tar.gz');assert sha(p)==h
        dest=stage/'cross'/key;cp=read(dest/'complete.json');z=load(dest/'features.pt');assert z['images']==list(by) and sha(dest/'features.pt')==cp['features']
        s,a=key.split('_');sig=read(stage/'inputs/lock.json')['starting'][s] if a=='parent' else read(stage/'models'/key/'complete.json')['final']
        assert z['adapter']==sig;pos={n:i for i,n in enumerate(z['images'])};seen=set()
        with tarfile.open(p,'r|gz') as tf:
            for m in tf:
                b=tf.extractfile(m).read()
                if m.name=='complete.json':assert json.loads(b)==cp;continue
                q=torch.load(io.BytesIO(b),weights_only=True);n=q['image'];r=by[n]
                assert sha_bytes(b)==cp['images'][n] and q['rgb']==r['rgb_sha256'] and q['query']==r['orig_query'] and q['adapter']==sig
                assert torch.equal(q['v'],z['features'][pos[n]]) and n not in seen;seen.add(n)
        assert len(seen)==448;out[key]=dict(images=len(seen),archive=h,adapter=sig)
    dump(stage/'results/cross_backup.json',out)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':main(*sys.argv[1:])
