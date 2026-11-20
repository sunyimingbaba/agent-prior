"""Bind the inputs and executable used in a new run before any group is scored."""
import sys
from shared import ROOT,read,sha,dump
group=sys.argv[1]
paths=['inputs/design.json','inputs/train.json','inputs/previous_coverage.csv','code/run.py','code/cases.py','code/shared.py','code/readouts.py']
if group=='fresh':
    paths+=['inputs/fresh.json','inputs/proposals.json','review/pixel_votes.json','review/画面重叠候选.csv']
file=ROOT/'inputs'/(group+'_lock.json')
d={p:sha(ROOT/p) for p in paths}
if file.is_file():assert read(file)==d
else:dump(file,d)
