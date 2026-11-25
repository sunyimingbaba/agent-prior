"""Serialize optional GPU work while allowing a long deployment sweep to yield between images."""
import fcntl
import time
from contextlib import contextmanager
from remote_common import ROOT,read
END=1795593060
def check_window():
    if time.time()>=END:raise SystemExit(75)
def priority_training():
    for stage,done in [('prefix','prefix_train_gpu'),('resolution','resolution_gpu')]:
        p=ROOT/'results'/(stage+'_nomination.json')
        if p.exists() and read(p)['eligible'] and not (ROOT/'logs'/(done+'.done')).exists():return True
    return False
@contextmanager
def lease():
    with (ROOT/'logs/optional_gpu.lock').open('a') as f:
        while True:
            check_window()
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(1)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
