"""Stop this experiment's optional readers after preserving raw traces at the requested end time."""
from pathlib import Path
import os,signal,time,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    assert time.time()>=1795593235
    readers={b'code/route_projection.py',b'code/swap_projection.py',b'code/cross_route.py',b'code/stochastic_deploy.py',b'code/learned_eval.py'}
    stopped=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if (p/'cwd').resolve()!=ROOT:continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            if not readers.intersection(args):continue
            if b'code/learned_eval.py' in args and b'project' not in args:continue
            pid=int(p.name);os.kill(pid,signal.SIGCONT);os.kill(pid,signal.SIGTERM)
            stopped.append(dict(pid=pid,entry=next(k.decode() for k in args if k in readers)))
        except (ProcessLookupError,FileNotFoundError,PermissionError):continue
    (ROOT/'results/window_close.json').write_text(json.dumps(dict(epoch=time.time(),optional_readers=stopped),indent=2)+'\n')
if __name__=='__main__':main()
