"""Paired changes in covered and unseen attributes on the same first-caption panels."""
import numpy as np
from score_final import ROOT, RFPO, read, paired, dump


def main():
    trained = {r["attribute"] for r in read(ROOT.parent/"cold/inputs/real_roster.json") if r["status"]=="selected"}
    per_image = read(ROOT/"results/逐图检索与细节.json")
    index = {(r["seed"],r["arm"],r["rgb"]):r["first106"] for r in per_image if "first106" in r}
    panel = [r for r in read(RFPO/"inputs/panel.json") if ("a","old3",r["rgb_sha256"]) in index]
    assert len(panel)==106 and len(trained)==8
    groups = {}
    for key, flag, expected in (("covered",True,20),("unseen",False,86)):
        rows = [r for r in panel if (r["attribute"] in trained)==flag]
        assert len(rows)==expected
        pairs = [r["pair"] for r in rows]
        vals = {arm:{s:{k:np.array([index[s,arm,r["rgb_sha256"]][k] for r in rows])
                        for k in ("DE","mrr")} for s in "ab"} for arm in ("old","old3","ordinary","retained")}
        groups[key] = {"images":len(rows),"pairs":len(set(pairs)), "attributes":sorted({r["attribute"] for r in rows}),
                       "old3_minus":{ref:paired(vals["old3"],vals[ref],pairs) for ref in ("old","ordinary","retained")}}
    dump(ROOT/"results/属性覆盖核对.json",groups)


if __name__ == "__main__":
    main()
