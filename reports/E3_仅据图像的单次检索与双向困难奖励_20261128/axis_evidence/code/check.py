from collections import Counter,defaultdict
from pathlib import Path
import io,hashlib,json
import numpy as np
import pyarrow as pa
import scipy
from PIL import Image
from scipy.stats import binomtest
from pixels import measure
from prepare import ROOT,REP,AK,sha,read,dump

def pack(groups):
    if not groups:return dict(pairs=0,images=0)
    pair=[];order=[];full=[];hits=[];counts=Counter()
    for g in groups:
        g=sorted(g,key=lambda a:a["label"])
        assert [r['label'] for r in g]==['horizontal','vertical']
        y=[r['hit'] for r in g];hits.extend(y);pair.append(np.mean(y));full.append(int(y==[1,1]))
        counts[str(sum(y))]+=1
        d=g[0]["margin"]-g[1]["margin"]
        order.append(0.5 if abs(d)<=1e-12 else float(d>0))
    def boot(x):
        a=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(281128)
        draws=rng.integers(len(a),size=(20000,len(a))); dist=a[draws].mean(1)
        return dict(mean=float(a.mean()),interval=np.quantile(dist,[.025,.975]).tolist())
    return dict(pairs=len(groups),images=len(hits),balanced_image_accuracy=boot(pair),pair_order_accuracy=boot(order),both_correct_pairs=sum(full),both_correct_fraction=boot(full),
                per_pair_hit_sum=dict(counts),image_correct=sum(hits),both_wrong_pairs=sum(v==0 for v in pair),
                swap_one_sided_exact_p=float(binomtest(sum(full),sum(z in (0,1) for z in pair),.5,alternative='greater').pvalue) if any(z in (0,1) for z in pair) else 1.)
def main():
    lock=read(ROOT/'inputs/lock.json')
    assert all(sha(ROOT/k)==v for k,v in lock['files'].items()) and all(sha(REP/k)==v for k,v in lock['sources'].items())
    raw=read(ROOT/'inputs/roster.json');tables={};digests={}; cache={}; by={};entries=[]
    for name,rs in raw.items():
        gs=defaultdict(list)
        for r in rs:
            sp=r["split"]
            if sp not in tables:
                p=Path('/Users/botlv/cup-deploy-assets/arrow/ucm')/('ucm_caption_karpathy_label_'+sp+'.arrow')
                tables[sp]=pa.ipc.open_file(pa.memory_map(str(p))).read_all();digests[sp]=sha(p)
            t=tables[sp];assert t['path'][r['row']].as_py()==r['image']
            im=Image.open(io.BytesIO(t['image'][r['row']].as_py())).convert('RGB'); rgb=np.asarray(im)
            assert hashlib.sha256(im.tobytes()).hexdigest()==r['rgb']
            if r['rgb'] not in cache:
                z=measure(rgb);rot=measure(np.rot90(rgb));error=abs(z['margin']+rot['margin'])
                assert error<=1e-10 and rot['axis']=={'tie':'tie','horizontal':'vertical','vertical':'horizontal'}[z['axis']]
                cache[r['rgb']]=dict(**z,rotation_error=error,rotated_axis=rot['axis'],rotated_margin=rot['margin'])
            z=cache[r['rgb']]
            row=dict(**r,**z,cohort=name,hit=.5 if z['axis']=='tie' else float(z['axis']==r['label']))
            gs[r["pair"]].append(row);entries.append(row)
        by[name]=list(gs.values())
    mirrors={s:{r['rgb'] for g in groups for r in g} for s,groups in by.items()}
    stats={k:{**pack(gs), 'by_attribute':{a:pack([g for g in gs if g[0]['attribute']==a]) for a in AK}} for k,gs in by.items()}
    distinct={}
    for name in by:
        other=set().union(*(mirrors[n] for n in by if n!=name))
        distinct[name]=pack([g for g in by[name] if not any(r['rgb'] in other for r in g)])
    pair_scores={name:[dict(pair=g[0]['pair'],attribute=g[0]['attribute'],results=[{k:r[k] for k in ('image','rgb','label','axis','margin','hit')} for r in g],
                        mean_image_accuracy=float(np.mean([r['hit'] for r in g]))) for g in groups] for name,groups in by.items()}
    out=dict(primary_nonoverlap=(lock['full_overlap']['train270__formal106']==0),summary=stats,
             slice_pixel_overlap={a+'__'+b:len(mirrors[a]&mirrors[b]) for i,a in enumerate(by) for b in list(by)[i+1:]},
             whole_roster_pixel_overlap=lock['full_overlap'],exclusive_full_pairs=distinct,total_unique_images=len(cache),
             all_90_degree_checks=True,max_rotation_error=max(x['rotation_error'] for x in cache.values()),arrow_sha256=digests,scipy=scipy.__version__,roster=sha(ROOT/'inputs/roster.json'))
    out['passes']=out['primary_nonoverlap'] and all(stats[n]['balanced_image_accuracy']['interval'][0]>.5 for n in ('train270','formal106'))
    dump(ROOT/'results/逐图.json',entries);dump(ROOT/'results/逐图对.json',pair_scores);dump(ROOT/'results/方向核验.json',out)
    print({k:(s['pairs'],s['balanced_image_accuracy'],s['both_correct_pairs']) for k,s in stats.items()},out['passes'],out['whole_roster_pixel_overlap'])
if __name__=='__main__':main()
