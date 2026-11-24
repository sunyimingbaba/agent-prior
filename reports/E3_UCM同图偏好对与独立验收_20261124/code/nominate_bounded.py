from dev_metrics import ROOT,read,dump,sha
def main():
    assert 'arm' not in read(ROOT/'inputs/nomination.json')
    p=ROOT/'results/dev_bounded.json';z=read(p)['difference_pp']['bounded-start'];eligible=min(z['CDE']['seeds'])>1e-8 and z['mR']['mean']>=-.5
    obj=dict(eligible=eligible,development=sha(p),previous=sha(ROOT/'inputs/nomination.json'),protocol=sha(ROOT/'名次约束续验口径.md'),panel=sha(ROOT/'confirm/inputs/panel.json'),CDE=z['CDE'],mR=z['mR'])
    if eligible:obj.update(arm='bounded',adapters={s:read(ROOT/'pilot'/(s+'_bounded')/'complete.json')['final'] for s in 'ab'})
    dest=ROOT/'inputs/nomination_bounded.json'
    if dest.exists():assert read(dest)==obj
    else:dump(dest,obj)
if __name__=='__main__':main()
