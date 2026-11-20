import csv,hashlib,json
from pathlib import Path
from shared import ROOT,read,dump
def clause(a,rel,b):
    return a.capitalize()+' is '+rel+' '+b+'.'
words={'left':'to the left of','right':'to the right of','above':'above','below':'below'}
opp={'left':'right','right':'left','above':'below','below':'above'}
def main():
    pool={r['image']:r for r in read(ROOT/'inputs/pool.json')}
    proposals=[];blind=[]
    with (ROOT/'inputs/landmarks.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            i=int(r['axis']=='y');xy=[[float(v) for v in r[k].split(',')] for k in ('xy_first','xy_second')]
            side=(['left','right'] if i==0 else ['above','below'])[int(xy[0][i]>xy[1][i])]
            rev=opp[side];a,b=r['first'],r['second']
            row={**pool[r['image']],'split':'val','scene_query':'An aerial view of the '+r['scene']+'.','landmarks':[a,b],'xy':xy,'axis':r['axis'],'direction':side,
                 'A':[clause(a,words[side],b),clause(a,words[rev],b)],'B':[clause(b,words[rev],a),clause(b,words[side],a)]}
            proposals.append(row);blind.append({'image':r['image'],'first':a,'second':b,'axis':r['axis']})
    dump(ROOT/'inputs/proposals.json',proposals);dump(ROOT/'review/blind_questions.json',blind)
if __name__=='__main__':main()
