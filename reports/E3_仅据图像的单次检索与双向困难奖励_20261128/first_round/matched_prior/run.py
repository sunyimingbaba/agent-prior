"""Hold pixels fixed while swapping the prior between the two members of a true opposite pair."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE = HERE.parents[1]
sys.path.insert(0,str(STAGE/"scene_evidence/code"))
import report as evaluation
sys.path.insert(0,str(STAGE/"audit"))
from image_inputs import tables, image, CONVERT, READOUT

def index_pairs(rows):
    pairs = defaultdict(list)
    for i,r in enumerate(rows):
        pairs[r["pair"]].append(i)
    source = np.empty(len(rows),dtype=int)
    for members in pairs.values():
        assert len(members)==2
        a,b=members
        assert rows[a]["scene_id"]==rows[b]["scene_id"] and rows[a]["side"]!=rows[b]["side"]
        source[a],source[b]=b,a
    assert np.array_equal(source[source],np.arange(len(rows)))
    return source

def frozen(reader,images,priors):
    vectors = []
    for i in range(0,len(images),4):
        vectors.append(reader(images[i:i+4],priors[i:i+4].float()).cpu())
    return torch.cat(vectors)

@torch.inference_mode()
def main():
    read,dump,sha=evaluation.read,evaluation.dump,evaluation.sha
    root=STAGE/"first_round/evaluation"; lock=read(root/"inputs/manifest.json")
    fn=READOUT/"cup_native_trace.pt"
    assert sha(fn)==read(READOUT/"proof.json")["trace"]
    reader=torch.jit.load(str(fn),map_location="cpu").eval()
    roster=read(evaluation.BASELINE/"inputs/roster.json")
    indexed={r["rgb"]:r for r in roster}; db=tables()
    official=read(evaluation.FORM/"inputs/ucm.json")
    ids={r["rgb_sha256"]:i for i,r in enumerate(official)}
    fine=[r for r in read(evaluation.FINE/"inputs/panel.json") if r["rgb_sha256"] in ids]
    extra=read(evaluation.OLD/"extra164/inputs/rows.json")
    text=torch.load(evaluation.FORM/"results/ucm_text.pt",weights_only=True)
    hard=torch.load(evaluation.FINE/"records/text.pt",weights_only=True)
    xt=torch.load(evaluation.OLD/"extra164/inputs/texts.pt",weights_only=True)
    cols=text["caption_order"].numpy().reshape(210,5)
    all_results={}
    for panel,rows in (("official",fine),("extra",extra)):
        counterpart=index_pairs(rows);labels=[r["pair"] for r in rows]
        name="first106" if panel=="official" else "extra164"
        summarize=evaluation.summary_by_seed if panel=="official" else evaluation.extra_stats
        pixels=torch.stack([CONVERT(image(indexed[r["rgb_sha256"]],db)) for r in rows])
        fullindex=[ids[r["rgb_sha256"]] for r in rows] if panel=="official" else list(range(len(rows)))
        groups=defaultdict(dict); checks={}
        zero=frozen(reader,pixels,torch.zeros(len(rows),3584))
        for seed in "ab":
            for arm in ("strict","r2"):
                tag=seed+"_"+arm; folder=root/"records"/tag/panel
                done=read(folder/"complete.json")
                assert done["adapter"]==lock["policies"][tag]["adapter"] and sha(folder/"features.pt")==done["features"]
                resources=torch.load(folder/"features.pt",map_location="cpu",weights_only=True)
                priors=[]
                for i,r in enumerate(rows):
                    file=folder/(r["image"]+".pt")
                    assert done["images"][r["image"]]==sha(file)
                    trace=torch.load(file,map_location="cpu",weights_only=True)
                    assert trace["rgb"]==r["rgb_sha256"] and trace["adapter"]==done["adapter"]
                    assert trace["query"]==trace["actual_query"]==lock["query"]
                    assert resources["images"][fullindex[i]]==r["image"]
                    priors.append(trace["prior"])
                priors=torch.stack(priors)
                own=frozen(reader,pixels,priors)
                swapped=frozen(reader,pixels,priors[counterpart])
                err=(own-resources["features"][fullindex]).abs().max().item()
                assert err<3e-4
                checks[tag]=dict(source=sha(folder/"complete.json"),max_readout_error=err)
                for cond,z in (("own",own),("opposite",swapped),("zero",zero)):
                    if panel=="official":
                        gallery=resources["features"].clone();gallery[fullindex]=z
                        scores=evaluation.details(gallery,fine,ids,cols,text,hard)
                    else:
                        scores=evaluation.metrics_extra(extra,xt,z)
                    groups[arm+"_"+cond][seed]={k:scores[k] for k in ("DE","mrr")}
        delta={}
        for arm in ("strict","r2"):
            for ref in ("opposite","zero"):
                delta[arm+"_own-"+ref]={s:{k:v-groups[arm+"_"+ref][s][k] for k,v in groups[arm+"_own"][s].items()} for s in "ab"}
        delta["interaction"]={s:{k:v-delta["r2_own-opposite"][s][k] for k,v in delta["strict_own-opposite"][s].items()} for s in "ab"}
        all_results[name]=dict(checks=checks,images=len(rows),pairs=len(set(labels)),levels={k:summarize(v,labels) for k,v in groups.items()},
                                     differences={k:summarize(v,labels) for k,v in delta.items()})
    all_results["pass"]=all(p["differences"]["strict_own-opposite"]["DE"]["interval"][0]>0
                            and p["differences"]["strict_own-opposite"]["mrr"]["mean"]>=0 for p in all_results.values())
    dump(HERE/"结果.json",all_results)
    print("matched-pair prior test",all_results["pass"])

if __name__=="__main__":
    torch.set_num_threads(3)
    main()
