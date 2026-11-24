"""Exploratory split by exposure to the exact scene and attribute in the 168 training images."""
import csv
import numpy as np
from candidate_guard import ROOT, OLD, read, dump, boot

def run():
    train = [r for sub in ('','confirm') for r in read(OLD/sub/'inputs/panel.json')]
    seen = {(r['scene_id'],r['attribute']) for r in train}
    first = read(ROOT/'confirm/inputs/panel.json')
    with (ROOT/'results/blind.csv').open(encoding='utf-8-sig') as f:
        scores={(r['tag'],r['image']):float(r['score']) for r in csv.DictReader(f) if r['metric']=='CDE'}
    second=read(ROOT/'reward_confirm/inputs/panel.json')
    scores.update({(r['seed']+'_'+r['arm'],r['image']):r['CDE'] for r in read(ROOT/'results/replica_rows.json')})
    report={}
    for name,cohort in [('first',first),('second',second)]:
        out={}
        for flag in (True,False):
            group=[r for r in cohort if ((r['scene_id'],r['attribute']) in seen)==flag]
            if not group:continue
            d=np.array([[scores[(s+'_bounded',r['image'])]-scores[(s+'_start',r['image'])] for r in group] for s in 'ab'])
            out['observed' if flag else 'new']=dict(images=len(group),difference=boot(d.mean(0),[r['pair'] for r in group]),seeds=d.mean(1).tolist())
        report[name]=out
    dump(ROOT/'results/coverage_exploratory.json',report)
if __name__=='__main__':run()
