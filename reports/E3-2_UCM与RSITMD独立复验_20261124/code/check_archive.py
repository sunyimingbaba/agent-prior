"""Verify the transferred archives and row payloads without extracting on the local machine."""
import hashlib,json,tarfile
from shared import ROOT,read,dump,sha
def main():
    index=read(ROOT/'records/index.json');count=0
    for name,sig in index['input_files'].items():assert sha(ROOT/'inputs'/name)==sig
    for name,sig in index['features'].items():assert sha(ROOT/'results'/name)==sig
    for name,sig in index['archives'].items():
        p=ROOT/'records'/name;assert sha(p)==sig
        state=name.removesuffix('_traces.tar.gz')
        with tarfile.open(p,'r:gz') as t:
            for dn in ('ucm','rsitmd'):
                base=f'{state}/{dn}/';proof=json.load(t.extractfile(base+'complete.json'))
                assert proof['adapter']==index['states'][state]['adapter']
                cases=read(ROOT/'inputs'/f'{dn}.json')
                assert len(cases)==len(proof['rows'])
                for c,h in zip(cases,proof['rows']):
                    f=t.extractfile(base+f"{c['row']:04d}.pt")
                    assert hashlib.sha256(f.read()).hexdigest()==h
                    count+=1
    dump(ROOT/'results/transfer_check.json',dict(rows=count,archives=len(index['archives']),feature_files=len(index['features'])))
if __name__=='__main__':main()
