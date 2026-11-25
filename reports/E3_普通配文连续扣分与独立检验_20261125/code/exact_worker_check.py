"""Replay three eligible training images and compare the hard-detail teacher with archived signed margins."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'source_transfer_2h_20261125/code'))
from remote_common import read,dump,sha
from runtime import BASE as LEGACY

def main():
    rows=read(BASE/'source_transfer_2h_20261125/inputs/training_roster.json')['rows']
    lookup={p.name:p for d in ('attribute_questions_20261125','attribute_questions_repeat_20261125') for p in (BASE/d/'records/a').glob('*.tif.pt')}
    inputs=[lookup[r['image']+'.pt'] for r in rows if r['image']+'.pt' in lookup][:3]
    worker=subprocess.Popen([str(LEGACY/'env/bin/python'),'-u',str(ROOT/'code/exact_worker.py')],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    report=[]
    try:
        assert worker.stdout.readline().strip()=='READY'
        for f in inputs:
            src=read(f.with_suffix('.json'));assert src['states']==sha(f)
            v=torch.load(f,weights_only=True)['scene']
            req=dict(image=src['image'],data=base64.b64encode(v.float().numpy().tobytes()).decode())
            worker.stdin.write(json.dumps(req)+'\n');worker.stdin.flush();resp=json.loads(worker.stdout.readline())
            old=src['groups']['scene']['scores']
            compare=[int(c['A']>0)+int(c['B']>0) for c in old]
            assert resp['detail'][:len(old)]==compare
            assert len(resp['rr'])==17 and all(len(vals)==len(resp['positive'])>0 for vals in resp['rr'])
            report.append(dict(image=src['image'],source=sha(f),detail_values_checked=len(old),held_captions=len(resp['positive'])))
    finally:
        worker.stdin.close();worker.wait()
    assert len(report)==3 and worker.returncode==0
    dump(ROOT/'results/exact_worker_check.json',dict(worker=sha(ROOT/'code/exact_worker.py'),groups=report))
if __name__=='__main__':main()
