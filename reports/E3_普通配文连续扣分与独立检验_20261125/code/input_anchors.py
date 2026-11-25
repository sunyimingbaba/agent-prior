"""Paired prompt-only change in the true single greedy prior for each candidate cohort."""
import numpy as np
from compare import OUT,read,dump,boot

def main():
    samples=[]
    for tag in ('fresh','extension','saved','extra'):
        prefix=('multi_'+tag) if tag in ('fresh','extension') else ('covered_'+tag+'_images')
        scene={(r['image'],r['seed']):r for r in read(OUT/'records'/(prefix+'.json'))}
        for r in read(OUT/'records'/('real_'+tag+'.json')):
            q=scene[r['image'],r['seed']];assert r['pair']==q['pair']
            samples.append((tag,r['pair'],q['means']['anchor'],r['means']['anchor']))
    report={}
    for key,tags in [('all',('fresh','extension','saved','extra')),('first138',('fresh','extension')),('other144',('saved','extra'))]:
        sub=[v for v in samples if v[0] in tags];labels=[a+'|'+b for a,b,_,_ in sub];result={}
        for i,k in [(1,'mrr'),(2,'detail')]:
            a=np.array([v[2][i] for v in sub]);b=np.array([v[3][i] for v in sub])
            result[k]=dict(scene=float(a.mean()),first_caption=float(b.mean()),change=boot(b-a,np.ones(len(a)),labels))
        report[key]=result
    dump(OUT/'results/input_anchors.json',report)
if __name__=='__main__':main()
