"""Locate the matched C/D/E decisions behind the warm-start final accuracy change."""
from collections import Counter
from compare import OUT,load,read,dump
from single_scores import RFPO,calculate
def main():
    rows=read(RFPO/'inputs/panel.json');records=[];counts={}
    for seed in 'ab':
        left=calculate('primary',load(OUT/'mined_policy'/(seed+'_selected_joint')/'evaluate/primary.pt')['features'],rows)
        right=calculate('primary',load(RFPO/'vectors'/(seed+'_deployed_primary.pt'))['features'],rows)
        for k in 'CDE':
            counts[seed+'_'+k]=dict(gained=int((left[k]>right[k]).sum()),lost=int((left[k]<right[k]).sum()))
            for i,r in enumerate(rows):
                if left[k][i]!=right[k][i]:records.append(dict(seed=seed,image=r['image'],attribute=r['attribute'],wording=k,pair=r[k],after=int(left[k][i])))
    improved=[r for r in records if r['after']];lost=[r for r in records if not r['after']]
    dump(OUT/'results/mined_detail_changes.json',dict(judgments=2*len(rows)*3,gained=len(improved),lost=len(lost),counts=counts,
        gained_by_attribute=dict(Counter(r['attribute'] for r in improved)),lost_by_attribute=dict(Counter(r['attribute'] for r in lost)),records=records))
if __name__=='__main__':main()
