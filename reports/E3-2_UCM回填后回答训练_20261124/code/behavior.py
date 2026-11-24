"""Measure post-tool behavior directly from native validation traces."""
import argparse,json
from pathlib import Path
import torch
from comparisons import ARMS,SEEDS,ROOT,write

def main(archive):
    summary={}
    for sub in SEEDS:
        result={}
        for arm in ARMS:
            files=sorted((archive/sub/arm/'eval').glob('*.pt'))
            assert len(files)==210
            entries=[torch.load(f,weights_only=True,map_location='cpu') for f in files]
            post=[x['rounds'][1] for x in entries if len(x['rounds'])>1]
            result[arm]=dict(images=len(entries),reached_second=len(post),end_on_first_token=sum(x['eos'] and x['length']==1 for x in post),
                mean_second_length=sum(x['length'] for x in post)/len(post) if post else None,
                reached_third=sum(len(x['rounds'])>2 for x in entries))
        summary[sub]=result
    write(ROOT/'results/after_tool_behavior.json',summary)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    main(parser.parse_args().archive)
