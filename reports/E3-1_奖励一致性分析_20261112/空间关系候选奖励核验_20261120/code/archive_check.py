"""Preserve the 32 original historical vectors for direct independent replay verification."""
import shutil
from pathlib import Path
import torch
from shared import ROOT,BASE,locked,read,sha,dump
out=ROOT/'archive_check'
out.mkdir(exist_ok=True)
verified=[]
for r in locked():
    src=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_train'/(str(r['row'])+'.pt')
    target=out/Path(r['image']).with_suffix('.pt')
    ref=ROOT/'samples/original'/target.stem/'greedy.pt'
    expected=read(ref.with_suffix('.json'))['history']['archive_sha']
    assert sha(src)==expected
    shutil.copyfile(src,target)
    assert sha(target)==expected and torch.equal(torch.load(target,weights_only=True,map_location='cpu').reshape(3584),
                                              torch.load(ref,weights_only=True,map_location='cpu'))
    verified.append(dict(image=r['image'],row=r['row'],source=str(src),sha256=expected,matched_replay=sha(ref)))
dump(out/'verified.json',verified)
