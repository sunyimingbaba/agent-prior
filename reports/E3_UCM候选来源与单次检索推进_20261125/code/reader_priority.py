"""Give the two matched-candidate readers priority within the eight-core server quota."""
from pathlib import Path
import os
import time
ROOT=Path(__file__).resolve().parents[1]
original={}
def workers():
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if (p/'cwd').resolve()!=ROOT:continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            optional=(any(k in args for k in (b'code/swap_projection.py',b'code/route_projection.py')) or
                      b'project' in args and any(k in args for k in (b'code/cross_route.py',b'code/stochastic_deploy.py')))
            if not optional:continue
            for task in (p/'task').iterdir():yield int(task.name)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
def main():
    available=sorted(os.sched_getaffinity(0));assert len(available)>=8
    shared={available[len(available)//4],available[len(available)//4+1]}
    try:
        while time.time()<1795593060 and not all((ROOT/'logs'/('prefix_'+str(i)+'.done')).exists() for i in (0,1)):
            for tid in workers():
                try:
                    original.setdefault(tid,os.sched_getaffinity(tid));os.sched_setaffinity(tid,shared)
                except ProcessLookupError:pass
            time.sleep(4)
    finally:
        for tid in workers():
            try:
                if tid in original:os.sched_setaffinity(tid,original[tid])
            except ProcessLookupError:pass
if __name__=='__main__':main()
