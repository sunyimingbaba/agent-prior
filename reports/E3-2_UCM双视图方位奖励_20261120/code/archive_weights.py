"""Store six final LoRA adapters as lossless deltas and verify each by restoring it."""
import argparse
from pathlib import Path
from shared import ROOT, sha, dump, read
from adapter_delta import main as delta

def archive(historical, selected=None):
    work=ROOT/'scratch/restore'
    work.mkdir(exist_ok=True,parents=True)
    arms=read(ROOT/'inputs/protocol.json')['arms']
    records={}
    for arm in ([selected] if selected else arms):
        assert arm in arms
        folder=ROOT/('policy_'+arm)
        original=folder/'step_16/adapter/adapter_model.safetensors'
        expected=read(folder/'update.json')['adapter_hash']
        assert sha(original)==expected
        bundle=ROOT/'adapters'/arm
        if not (bundle/'index.json').exists(): delta('pack',historical,original,bundle)
        dest=work/(arm+'.safetensors')
        if not dest.exists(): delta('restore',historical,bundle,dest)
        assert sha(dest)==expected and dest.stat().st_size==original.stat().st_size
        index=read(bundle/'index.json')
        assert index['output_sha']==expected and index['base_sha']==sha(historical)
        records[arm]=dict(archive='adapters/'+arm, restored_sha256=expected, bytes=dest.stat().st_size,
                          trained_parameters=read(folder/'update.json')['final_params'],parts=len(index['parts']))
    if not selected:
        dump(ROOT/'results/weight_restoration.json',dict(base_sha256=sha(historical),checkpoints=records))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('base',type=Path);parser.add_argument('--arm')
    args=parser.parse_args();archive(args.base,args.arm)
