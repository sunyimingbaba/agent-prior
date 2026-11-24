"""Archive and authenticate each run, and count actual tool invocations and second-round EOS."""
import io,sys,tarfile
from collections import Counter
from pathlib import Path
import torch
from shared import ASSETS,ROOT,BASE,read,dump,sha
from data import DOMAINS,PREVIOUS,STATES

def main():
    dst=ROOT/'records';dst.mkdir(exist_ok=True)
    eos=read(ASSETS/'models/Qwen2.5-VL-7B-Instruct/generation_config.json')['eos_token_id']
    stats={};hashes={}
    for arm,(sub,name) in STATES.items():
        cp=PREVIOUS/sub/name/'adapter/adapter_model.safetensors';weight=sha(cp)
        groups={}
        for dn in DOMAINS:
            src=ROOT/'outputs'/arm/dn;proof=read(src/'complete.json');cases=read(ROOT/'inputs'/f'{dn}.json')
            assert proof['adapter']==weight and len(proof['rows'])==len(cases)
            counts=Counter()
            for c,sig in zip(cases,proof['rows']):
                f=src/f"{c['row']:04d}.pt"
                assert sha(f)==sig
                obj=torch.load(f,map_location='cpu',weights_only=True)
                assert obj['adapter']==weight and obj['rgb']==c['rgb_sha256'] and obj['image']==c['image'] and obj['query']==c['orig_query']
                rounds=obj['rounds']
                assert len(rounds) in (1,2,3) and obj['prior'].shape==(3584,) and obj['prior'].dtype==torch.bfloat16
                counts['images']+=1;counts['tool_called']+=bool(obj['tools']);counts['second_round']+=len(rounds)>=2
                counts['second_immediate_eos']+=len(rounds)>=2 and len(rounds[1]['ids'])==1 and rounds[1]['ids'][0] in eos
                counts['third_round']+=len(rounds)==3
            groups[dn]=dict(counts)
        pack=dst/(arm+'_traces.tar.gz')
        with tarfile.open(pack,'w:gz',compresslevel=3) as t:t.add(ROOT/'outputs'/arm,arcname=arm)
        hashes[pack.name]=sha(pack)
        stats[arm]=dict(seed=read(PREVIOUS/sub/'initial.json')['seed'],adapter=weight,groups=groups)
    source={str(f):sha(f) for f in (BASE/'CUP/agent/rollout.py',BASE/'CUP/agent/tools.py',
        BASE/'CUP/vilt/modules/vilt_module.py',ASSETS/'ucm_main_domain_tool.ckpt',ASSETS/'ucm_scene_cls.pt',
        ASSETS/'models/ViT-L-14.pt',ASSETS/'models/Qwen2.5-VL-7B-Instruct/generation_config.json')}
    result_hashes={p.name:sha(p) for p in (ROOT/'results').glob('*.pt')}
    assert len(result_hashes)==17
    dump(dst/'index.json',dict(states=stats,archives=hashes,features=result_hashes,sources=source,
         environment=read(ROOT/'environment.json'),input_files={p.name:sha(p) for p in (ROOT/'inputs').glob('*.json')}))
if __name__=='__main__':main()
