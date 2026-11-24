from hashlib import sha256
from report_io import ROOT,read,dump,sha,trainrows

if __name__=='__main__':
    roster=read(ROOT/'inputs/roster.json'); rows=trainrows()
    pairs={rows[n]['pair'] for n in roster['order']}
    selected=set(sorted(pairs,key=lambda p:sha256(('message-role-v1|'+p).encode()).digest())[:12])
    names=[n for n in roster['order'] if rows[n]['pair'] in selected]
    assert len(names)==24 and all(sum(rows[n]['pair']==p for n in names)==2 for p in selected)
    dump(ROOT/'inputs/message_lock.json',dict(seed='a',images=names,pairs={n:rows[n]['pair'] for n in names},
        roster=sha(ROOT/'inputs/roster.json'),protocol=sha(ROOT/'工具回填对照口径.md')))
