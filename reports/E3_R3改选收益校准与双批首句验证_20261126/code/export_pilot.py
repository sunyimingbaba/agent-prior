"""Export numeric parameters and the exact selector functions used in the CPU scoring worker."""
import ast
import inspect
import json
from pathlib import Path
import textwrap
import joblib
import numpy as np
from data import ROOT, PREV, read, stream, best, centered, gains_features

OUT=ROOT/'inputs/pilot'


def extract(path,names):
    source=path.read_text()
    return '\n\n'.join(ast.get_source_segment(source,node) for node in ast.parse(source).body
                        if isinstance(node,ast.FunctionDef) and node.name in names)


def main():
    sources=[textwrap.dedent(inspect.getsource(fn)) for fn in (best,centered,gains_features)]
    sources.extend(extract(ROOT/'code'/file,names) for file,names in [
        ('study.py',{'choose','old_choices'}),('neighbor.py',{'unit','quantiles_of_scores','stats_matrix','decide'})])
    (OUT/'pilot_rules.py').write_text('import numpy as np\nEPS=1e-10\nTINY=1e-10\n\n'+'\n\n'.join(sources)+'\n')
    model=joblib.load(ROOT/'inputs/switch_model.joblib')
    forest=model['trees']
    trees={}
    for j,t in enumerate(forest.estimators_):
        n=t.tree_
        for k,a in [('left',n.children_left),('right',n.children_right),('feature',n.feature),('threshold',n.threshold),('v',n.value[:,:,0])]:
            trees[str(j)+'_'+k]=a
    np.savez_compressed(OUT/'trees.npz',**trees)
    old=read(PREV/'锁定评分参数.json')['model']
    nominated=read(ROOT/'inputs/提名.json');local=read(ROOT/'inputs/配文预测提名.json')
    cfg=dict(tree_count=len(forest.estimators_), switch={k:nominated[k] for k in ('family','weight','mode')},
             old=(np.array(old['coef'])/np.array(old['scale_x'])[:,None]).tolist(),
             neighbor={k:local[k] for k in ('k','mode','weight','coefficients')},
             ridge={k:a.tolist() for k,a in model['ridge'].items()})
    (OUT/'reward.json').write_text(json.dumps(cfg,indent=2)+'\n')
    data=read(ROOT/'inputs/pilot/train.json')
    byname={r['image']:r['pair'] for rows in data.values() for r in rows}
    (OUT/'pair_by_name.json').write_text(json.dumps(byname,indent=2)+'\n')
    for seed in 'ab':
        pairs=[];names=[];text=[];image=[];captions=[];spans=[0]
        for r,s,v,b,vis in stream('dev'):
            if s!=seed or v!='first':continue
            ids=sorted(set(b['maps'][r['image']]['native']))
            ts=b['native'][ids].copy()
            center=ts.mean(0);center/=np.linalg.norm(center)
            pairs.append(r['pair']);names.append(r['image']);text.append(center);image.append(vis[-1]);captions.extend(ts);spans.append(len(captions))
        np.savez_compressed(OUT/(seed+'_prototypes.npz'),pairs=pairs,names=names,text=text,image=image,captions=captions,spans=spans)


if __name__=='__main__':main()
