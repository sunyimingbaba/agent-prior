"""Compare every final adapter tensor with its recoverable optimizer checkpoint."""
import hashlib
import json
from pathlib import Path
import sys
from safetensors import safe_open
import torch
ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main(name):
    dest=ROOT/'models'/name
    proof=read(dest/'complete.json')
    receipt=ROOT/'inputs/pilot/decision.json'
    seed,arm=name.split('_')
    assert proof['lock']==sha(receipt) and proof['initial']==read(receipt)['starts'][seed]
    resume=dest/'resume.pt'
    z=torch.load(resume,map_location='cpu',weights_only=True,mmap=True)
    assert z['lock']==proof['lock'] and z['history']==read(dest/'steps.json') and proof['scores']==sha(dest/'steps.json')
    assert len(z['history'])==proof['steps']==13 and proof['nonzero']==sum(r['norm']>0 for r in z['history'])
    assert len(z['opt']['param_groups'])==1
    steps={int(v['step']) for v in z['opt']['state'].values()}
    assert steps=={proof['nonzero']}
    adapter=dest/'adapter/adapter_model.safetensors'
    assert sha(adapter)==proof['final']
    with safe_open(str(adapter),framework='pt') as handle:
        keys={k.replace('.default.','.'):k for k in z['weights']}
        assert set(keys)==set(handle.keys())
        for k in keys:
            assert torch.equal(z['weights'][keys[k]],handle.get_tensor(k))
    result={'receipt':z['lock'],'resume_sha256':sha(resume),'final_sha256':proof['final'],'identical_tensors':len(keys),
            'history_steps':proof['steps'],'optimizer_updates':sorted(steps),'optimizer_states':len(z['opt']['state'])}
    (ROOT/'results'/('resume_'+name+'.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1)
    main(sys.argv[1])
