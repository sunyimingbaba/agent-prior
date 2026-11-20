"""Compare locally regenerated statistics with the independently computed private-server files."""
import sys
from pathlib import Path
from shared import ROOT,sha,dump
def main(server):
    result=ROOT/'results'
    records={}
    for file in sorted(server.iterdir()):
        if file.is_file() and file.suffix in ('.csv','.json') and file.name!='local_recompute.json':
            local=result/file.name
            digest=sha(local);assert digest==sha(file),file.name
            records[file.name]=digest
    assert set(records)=={p.name for p in result.iterdir() if p.is_file() and p.name!='local_recompute.json'}
    dump(result/'local_recompute.json',dict(byte_identical_outputs=len(records),hashes=records))
if __name__=='__main__':main(Path(sys.argv[1]))
