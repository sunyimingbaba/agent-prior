from pathlib import Path
from compare import OUT,SOURCE,REP,read,dump,sha
def main():
    q={}
    for p in REP.glob('E3_UCM中性观察问题*/inputs/lock.json'):
        for j,r in enumerate(read(p)['rows']):q[r['image']]=j
    fb=read(OUT/'inputs/feedback.json');roster=[]
    for tag in ('fresh','extension','saved','extra'):
        p=OUT/'inputs'/((tag if tag in ('fresh','extension') else 'covered_'+tag)+'.json')
        for row in read(p):
            c=dict(row);n=c['image']
            if tag in ('fresh','extension'):base=31000000+c['row']*1000
            elif tag=='extra':base=51000000+c['row']*1000
            else:base=10000000+q[n]*1000
            roster.append(dict(c,tag=tag,rng_a=base,rng_b=base+10000000,feedback=c.get('feedback',fb.get(n,[None])[0])))
    assert len(roster)==len({r['rgb_sha256'] for r in roster})==282 and all(r['feedback'] for r in roster)
    p=OUT/'inputs/real_roster.json';dump(p,roster);dump(OUT/'inputs/real_lock.json',dict(roster=sha(p),protocol=sha(OUT/'原首句真实输入的候选复核.md'),reward=sha(OUT/'inputs/multi_nomination.json')))
if __name__=='__main__':main()
