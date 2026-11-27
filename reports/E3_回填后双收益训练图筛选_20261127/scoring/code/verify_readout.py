"""Check native float readouts with the historical four-thread setting."""
from pool_reader import ORIGINAL, REBUILD, SCORING, STAGE, Reader, grade, load, read, save
import hashlib
import sys
import numpy as np
import torch

@torch.inference_mode()
def main():
    reader=Reader()
    torch.set_num_threads(4)
    native=load(REBUILD/"relation_candidates_20261120/native/features.pt")
    idx=read(REBUILD/"relation_candidates_20261120/native/index.json")
    even=np.array([int(hashlib.sha256(",".join(map(str,q)).encode()).hexdigest()[-1],16)%2==0 for q in native["tokens"].tolist()])
    cases = ["17.tif","54.tif","60.tif","80.tif"]
    verified={}
    for name in cases:
        raw=load(STAGE/"sampling/raw/a"/(name+".pt"))
        prev=read(STAGE/"scores/a"/(name+".json"))
        row=reader.rows[name]
        pos=[idx["columns"][q] for q in row["captions"]]
        pos_alt=sorted(set(pos)-{pos[0]})
        assert pos_alt==row["alt_ids"]
        negative=even.copy(); held=~even
        negative[pos]=held[pos]=False
        pixels=reader.native.preprocess(reader.rgb(reader.images,row))[None]
        results=[]
        for state in raw["hidden"].float():
            rank=reader.proxy.score(state.tolist(),row["row"])[0]
            vec=reader.native.img_embeds(pixels,agent_hidden=state[None])["image_feats"][0].double()
            sims=native["features"].double()@vec
            hs=reader.hard[name].double()@vec
            ds=(hs[::2]-hs[1::2]).tolist()
            other=[]
            for col in pos_alt:
                higher=int((sims[held]>sims[col]).sum()); tied=int((sims[held]==sims[col]).sum())
                other.append(float(np.mean(1/np.arange(higher+1,higher+tied+2))))
            results.append(dict(rank=int(rank),margin=float(sims[pos[0]]-sims[negative].max()),
                                hard_margins=ds[:2],DE=ds[2:4],FG=ds[4:],other_caption_rr=other))
        alt=grade(results,reader.match,reader.bonus)
        verified[name]=dict(n=8,identical=results==prev["items"],
            maximum_margin_delta=max(abs(x["margin"]-y["margin"]) for x,y in zip(results,prev["items"])),
            max_hard_delta=max(abs(j-k) for x,y in zip(results,prev["items"])
                               for j,k in zip(x["hard_margins"],y["hard_margins"])),
            same_rank=all(x["rank"]==y["rank"] for x,y in zip(results,prev["items"])),
            same_rr=all(x["other_caption_rr"]==y["other_caption_rr"] for x,y in zip(results,prev["items"])),
            same_r2=alt["r2"]==prev["r2"],same_bonus=alt["bonus"]==prev["bonus"],
            same_opportunities=alt["opportunity_slots"]==prev["opportunity_slots"],
            same_r3_top=alt["r3_top"]==prev["r3_top"])
    save(SCORING/"results/readout_check.json",verified)
    print(verified)


if __name__=="__main__":
    main()
