"""Rotate screen-axis relations clockwise, preserving subjects, references and word order."""
import re
from PIL import Image
from shared import ROOT,read,sha,dump
directions={'above':'to the right of','to the right of':'below','below':'to the left of','to the left of':'above'}
pattern=re.compile(r' (above|below|to the left of|to the right of) ')
def clockwise(s):
    result,n=pattern.subn(lambda m:' '+directions[m.group(1)]+' ',s)
    assert n==1
    return result
def check_geometry():
    for label,(dx,dy) in {'above':(0,-1),'below':(0,1),'to the left of':(-1,0),'to the right of':(1,0)}.items():
        pic=Image.new('L',(5,5));pic.putpixel((2+dx,2+dy),255)
        right=pic.transpose(Image.Transpose.ROTATE_270);left=pic.transpose(Image.Transpose.ROTATE_90)
        assert right.getpixel((2-dy,2+dx))==left.getpixel((2+dy,2-dx))==255
        actual='to the right of' if -dy==1 else 'to the left of' if -dy==-1 else 'below' if dx==1 else 'above'
        assert actual==directions[label]
check_geometry()
source=ROOT/'inputs/detail.json';rows=read(source)
for r in rows:
    r['direction_original']=r['direction']
    r['direction']={'above':'right','right':'below','below':'left','left':'above'}[r['direction_original']]
    for w in ('C','D'):
        for a,b in [r[w],r[w][::-1]]:
            assert clockwise(clockwise(a))==b
        r[w]=list(map(clockwise,r[w]))
output=ROOT/'inputs/quarter_questions.json';dump(output,rows)
dump(ROOT/'inputs/quarter_protocol.json',dict(original=sha(source),questions=sha(output),scene_cue_only=True,views=['cw90','ccw90'],
    transform='PIL ROTATE_270 is clockwise. cw-positive text rotates only the screen-relative predicate; ccw-positive swaps cw alternatives. The source image, subject/reference and two word orders are identical.',
    endpoint='Supplementary only, all six final checkpoints and history regardless of outcome. Score both opposite quarter turns, and intersection of both quarter and both primary views per source and wording. These are repeated views of the same 16 validation photos. Training uses 0 and 180 degrees only.'))
