"""Use the frozen on-policy readouts for both candidate screening and training."""
import argparse
import ast
import base64
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

SCORING = Path(__file__).resolve().parents[1]
STAGE = SCORING.parent
REBUILD = STAGE.parent
ORIGINAL = REBUILD/"second_answer_5090_20261124/code"
BONUS = REBUILD/"graded_r2_detail_20261127/code/detail_reward.py"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    part = path.with_name(path.name+".part")
    part.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False)+"\n")
    part.replace(path)


def load(path):
    return torch.load(path,map_location="cpu",weights_only=True)


def source_func(path,name,expected):
    assert sha(path) == expected, path
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n,"name",None) == name)
    scope = {"np":np}
    exec(compile(ast.Module([node],type_ignores=[]),str(path),"exec"),scope)
    return scope[name]


def opportunity(detail,other,top):
    if not other[0]:
        return []
    # Exact comparison of sums of the available reciprocal ranks. Equal
    # candidates stay equal even for non-power-of-two top-set sizes.
    sums = [sum(Fraction.from_float(v) for v in row) for row in other]
    target = sum(sums[j] for j in top)
    acc_numerator = sum(detail[j] for j in top)
    return [i for i in range(8) if detail[i]*len(top)>acc_numerator and sums[i]*len(top)>=target]


def grade(items,match,bonus_fn):
    margin = np.asarray([x["margin"] for x in items])
    rank = np.asarray([x["rank"] for x in items])
    hard = np.asarray([x["hard_margins"] for x in items])
    r2 = match(margin,1/(1+rank))
    bonus = bonus_fn(r2,hard)
    r3 = r2+bonus
    new = np.flatnonzero(r3==r3.max()).tolist()
    top = np.flatnonzero(r2==r2.max()).tolist()
    other = [x["other_caption_rr"] for x in items]
    slots = opportunity((hard>0).sum(1).tolist(),other,top)
    return dict(r2=r2.tolist(),r3=r3.tolist(),bonus=bonus.tolist(),r2_top=top,r3_top=new,
                opportunity_slots=slots,achieved=bool(set(new)&set(slots)))


class Reader:
    def __init__(self):
        manifest = read(SCORING/"inputs/manifest.json")
        for name,expected in manifest["compiled"].items():
            assert sha(SCORING/"inputs"/name) == expected, name
        assert sha(ORIGINAL/"readouts.py") == manifest["readouts_source"]
        assert sha(ORIGINAL/"shared.py") == manifest["shared_source"]
        sys.path.insert(0,str(ORIGINAL))
        from shared import rgb, setup, table
        setup()
        torch.set_num_threads(4)
        from readouts import load_model, scorer_for
        self.match = source_func(ORIGINAL/"train.py","matched_scores",manifest["training_source"])
        self.bonus = source_func(BONUS,"graded_bonus",manifest["bonus_source"])
        relation = REBUILD/"relation_candidates_20261120"
        gallery = relation/"gallery/train.pt"
        bankpath = relation/"native/features.pt"
        assert sha(gallery) == manifest["gallery"] and sha(bankpath)==manifest["native"]
        bank = load(bankpath)
        self.features = bank["features"].double()
        self.even = np.array([int(hashlib.sha256(",".join(map(str,q)).encode()).hexdigest()[-1],16)%2==0
                              for q in bank["tokens"].tolist()])
        self.hard = load(SCORING/"inputs/vectors.pt")
        self.rows = {r["image"]:r for r in read(SCORING/"inputs/rows.json")}
        self.native = load_model(False)
        self.proxy = scorer_for(load_model(True),gallery)
        self.images = table("train")
        self.rgb = rgb

    @torch.inference_mode()
    def score(self,name,states):
        assert states.shape == (8,3584) and torch.isfinite(states).all()
        case = self.rows[name]
        assert case["captions"] == self.images["caption"][case["row"]].as_py()
        pixels = self.native.preprocess(self.rgb(self.images,case))[None]
        neg = self.even.copy()
        alt_neg = ~self.even
        pos = case["native_ids"]
        neg[pos] = alt_neg[pos] = False
        split = torch.from_numpy(neg)
        alt_split = torch.from_numpy(alt_neg)
        hard = self.hard[name].double()
        items,cache = [],{}
        for state in states.float().contiguous():
            key = state.numpy().tobytes()
            if key not in cache:
                rank = self.proxy.score(state.tolist(),case["row"])[0]
                v = self.native.img_embeds(pixels,agent_hidden=state[None])["image_feats"][0].double()
                sims = self.features @ v
                margin = float(sims[pos[0]]-sims[split].max())
                other = []
                negatives = sims[alt_split]
                for col in case["alt_ids"]:
                    higher = int((negatives>sims[col]).sum())
                    same = int((negatives==sims[col]).sum())
                    other.append(float(np.mean(1/np.arange(higher+1,higher+same+2))))
                scores = hard@v
                signed = (scores[::2]-scores[1::2]).tolist()
                cache[key] = dict(rank=int(rank),margin=margin,hard_margins=signed[:2],
                                  DE=signed[2:4],FG=signed[4:] or None,other_caption_rr=other)
            items.append(cache[key])
        return {"items":items,**grade(items,self.match,self.bonus)}


def service():
    out = os.fdopen(os.dup(1),"w",buffering=1)
    os.dup2(2,1)
    reader = Reader()
    print("READY",file=out)
    for line in sys.stdin:
        msg = json.loads(line)
        states = np.frombuffer(base64.b64decode(msg["hidden"],validate=True),dtype="<f4").copy().reshape(8,3584)
        result = reader.score(msg["image"],torch.from_numpy(states))
        print(json.dumps(result,allow_nan=False),file=out)


def score_files(args):
    if args.seed == "b":
        for name in ("a_selection","training"):
            path = SCORING/"locks"/(name+".json")
            assert sha(path) == (SCORING/"locks"/(name+".sha256")).read_text().strip()
    reader = Reader()
    for i,(name,case) in enumerate(reader.rows.items()):
        if i%args.nshards != args.shard:
            continue
        root = STAGE/"sampling"
        p = root/"raw"/args.seed/(name+".pt")
        out = STAGE/"scores"/args.seed/(name+".json")
        while not p.is_file():
            time.sleep(2)
        if out.exists():
            previous = read(out)
            assert previous["source"] == sha(p) and previous["rgb"] == case["rgb_sha256"]
            continue
        raw = load(p)
        assert raw["image"]==name and raw["row"]==case["row"] and raw["rgb"]==case["rgb_sha256"]
        assert raw["seed"]==args.seed
        assert raw["query"]==case["orig_query"]
        eligible = raw["eligible"]
        result = dict(image=name,cohort=case["cohort"],pair=case["pair"],rgb=raw["rgb"],
                      eligible=eligible,alt_ids=case["alt_ids"],source=sha(p),seed=args.seed)
        if eligible:
            assert raw["feedback"] == case["feedback"]
            result.update(reader.score(name,raw["hidden"]))
        save(out,result)
        print(i,name,eligible,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--service",action="store_true")
    parser.add_argument("--seed",choices=["a","b"])
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--nshards",type=int,default=1)
    args = parser.parse_args()
    if args.service:
        service()
    else:
        assert args.seed
        score_files(args)
