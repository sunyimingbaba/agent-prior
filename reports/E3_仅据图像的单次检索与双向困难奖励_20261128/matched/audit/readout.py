"""Read the first and last actual group of each completed arm with the historical CPU scorer."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
POOL=ROOT.parents[1]/"aftertool_joint_pool_20261127"
sys.path.insert(0,str(POOL/"scoring/code"))
from pool_reader import Reader,sha,read,load,save


def main():
    model=Reader()
    assert torch.get_num_threads()==4
    check={}
    for seed in "ab":
        for arm in ("r2","strict"):
            tag=f"{seed}_{arm}"
            folder=ROOT/"models"/tag
            if (folder/"no_eligible.json").exists():continue
            rows=read(folder/"steps.json");meta=read(folder/"complete.json")
            assert meta["scores"]==sha(folder/"steps.json")
            verified=[]
            for idx in sorted({0,len(rows)-1}):
                step=rows[idx]
                name=step["image"]
                path=folder/"trajectories"/(name+".pt")
                assert step["trajectory"]==sha(path)
                record=load(path)
                rerun=model.score(name,record["terminal_hidden"])
                assert rerun["r2"]==step["retrieval"]
                assert [v["hard_margins"] for v in rerun["items"]]==step["hard"]
                assert [v["rank"] for v in rerun["items"]]==step["proxy_rank"]
                assert [v["margin"] for v in rerun["items"]]==step["native_margin"]
                assert [v["other_caption_rr"] for v in rerun["items"]]==step["other_caption_rr"]
                verified.append(dict(image=name,slot=step["step"],trajectory=sha(path)))
            check[tag]=verified
    save(ROOT/"audit/末端状态原生读数抽验.json",dict(groups=check,
         frozen_reader=sha(POOL/"scoring/code/pool_reader.py"),code=sha(Path(__file__)),threads=torch.get_num_threads()))
if __name__=="__main__":main()
