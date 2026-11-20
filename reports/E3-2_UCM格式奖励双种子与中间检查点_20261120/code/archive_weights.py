"""Restore every Git-portable final checkpoint and verify it against its full saved adapter."""
import argparse
from pathlib import Path
from shared import ROOT,sha,dump,read
from adapter_delta import main as delta

def main(historical,previous):
    clipped={}
    work=ROOT/'scratch/restore';work.mkdir(parents=True,exist_ok=True)
    for arm in read(ROOT/'inputs/protocol.json')['arms']:
        folder=ROOT/('policy_'+arm)
        actual=folder/'step_24/adapter/adapter_model.safetensors'
        expected=read(folder/'update.json')['adapter_hash'];assert sha(actual)==expected
        previous_index=read(previous/'adapters/hard/index.json')
        if previous_index['output_sha']==expected:
            archive=previous/'adapters/hard';storage='previous_stage/adapters/hard'
        else:
            archive=ROOT/'adapters'/arm;storage='adapters/'+arm
            if not (archive/'index.json').is_file():
                delta('pack',historical,actual,archive)
        dest=work/(arm+'.safetensors')
        if not dest.exists():
            delta('restore',historical,archive,dest)
        assert sha(dest)==expected and dest.stat().st_size==actual.stat().st_size
        clipped[arm]=dict(storage=storage,output_sha256=expected,bytes=actual.stat().st_size,
                          trained_parameter_digest=read(folder/'update.json')['final_params'],restored_sha256=sha(dest))
    dump(ROOT/'results/weight_restoration.json',clipped)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('historical',type=Path);parser.add_argument('previous',type=Path)
    args=parser.parse_args();main(args.historical,args.previous)
