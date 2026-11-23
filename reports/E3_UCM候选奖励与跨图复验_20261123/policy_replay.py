"""Re-evaluate actual single-generation policies from the historical same-start trial."""
import numpy as np

from experiment import OUT, REPORTS, read, save, sha, table, tensor
from gallery_replay import feature_scores, per_source


def main():
    previous=REPORTS/"E3-2_UCM同起点三奖励短训_20261120"
    results=[]
    stats={}
    queries=np.random.default_rng(20261123).multinomial(210,np.full(210,1/210),size=20000)
    for cue in ["native","generic"]:
        arms={}
        for arm in ["base","old","caption","hard"]:
            p=previous/"eval"/arm
            proof=read(p/"scored.json")
            assert sha(p/"vectors.pt")==proof["vectors"] and sha(p/"text.pt")==proof["text"]
            pack=tensor(p/"vectors.pt")[cue]
            text=tensor(p/"text.pt")
            order=np.asarray(text["caption_order"])
            assert len(set(pack["order"]))==len(pack["order"])==210
            assert all(view=="upright" and k==-1 for _,view,k in pack["order"])
            if arm=="base":
                ids=pack["order"]
            else:
                assert pack["order"]==ids
            ss=feature_scores(pack["visual"].double().numpy(),text["native"].double().numpy())[:,order]
            owner=np.repeat(np.arange(210),5)
            first=set(order.reshape(210,5)[:,0])
            for gallery,keep in [("full",np.ones(1050,bool)),("exclude_first_globally",~np.isin(order,list(first)))]:
                v,d=per_source(ss,keep,owner)
                vals=v.sum(0)/d.sum(0)
                arms[(arm,gallery)]=v,d
                results.append(dict(cue=cue,arm=arm,gallery=gallery,query_images=int(d[:,0].sum()),
                                    captions=int(d[:,3].sum()),mR=vals.mean(),
                                    **dict(zip(["i1","i5","i10","t1","t5","t10"],vals,strict=True))))
        for arm,ref in [("caption","old"),("hard","old"),("hard","caption"),("caption","base"),("old","base")]:
            for gallery in ["full","exclude_first_globally"]:
                v,d=arms[arm,gallery]
                a=v-arms[ref,gallery][0]
                boot=((queries@a)/(queries@d)).mean(1)
                stats[f"{cue}/{gallery}/{arm}-{ref}"]=dict(delta=(a.sum(0)/d.sum(0)).mean(),
                                                    lo=np.quantile(boot,.025),hi=np.quantile(boot,.975))
    historical=[r for r in results if r["cue"]=="native" and r["gallery"]=="full"]
    for r,expected in zip(historical,[.6006,.4906,.4825,.5016],strict=True):
        assert abs(r["mR"]-expected)<.0001
    table("历史短训策略单次生成复验.csv",results)
    save("历史短训策略配对差.json",stats)


if __name__=="__main__":
    main()
