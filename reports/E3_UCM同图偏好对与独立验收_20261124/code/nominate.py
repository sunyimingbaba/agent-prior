"""Choose on development data only. Leave the blind outcome unopened if none qualifies."""
from dev_metrics import ROOT,read,dump,sha
def main():
    candidates=[];sources={}
    for label in ('observed','continuous'):
        file=ROOT/'results'/('dev_'+label+'.json');z=read(file);sources[label]=sha(file)
        for k,v in z['difference_pp'].items():
            if not k.endswith('-start'):continue
            arm=k[:-6];d=v['CDE'];n=v['mR'];eligible=min(d['seeds'])>1e-8 and n['mean']>=-.5
            candidates.append(dict(arm=arm,CDE=d['mean'],mR=n['mean'],per_seed=d['seeds'],eligible=eligible))
    order=sorted((c for c in candidates if c['eligible']),key=lambda c:(-c['CDE'],-c['mR'],c['arm']))
    obj=dict(candidates=candidates,sources=sources,protocol=sha(ROOT/'验收与比较约定.md'),panel=sha(ROOT/'confirm/inputs/panel.json'))
    if order:
        arm=order[0]['arm'];obj.update(arm=arm,adapters={s:read(ROOT/'pilot'/(s+'_'+arm)/'complete.json')['final'] for s in 'ab'})
    path=ROOT/'inputs/nomination.json'
    if path.exists():assert read(path)==obj
    else:dump(path,obj)
if __name__=='__main__':main()
