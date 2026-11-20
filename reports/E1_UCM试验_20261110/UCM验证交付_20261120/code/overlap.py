"""Screen same-scene UCM train/validation tiles for shared RGB content. No scores used."""
import csv
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import pyarrow as pa
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(1)
data=Path(sys.argv[1])
scored={r['image'] for r in json.loads((ROOT/'inputs/cases.json').read_bytes())}
names={}
descriptors={}
for split in ('train','val'):
    tbl=pa.ipc.open_file(str(data/f'ucm_caption_karpathy_label_{split}.arrow')).read_all()
    sift=cv2.SIFT_create(nfeatures=240)
    for n,im in zip(tbl['path'].to_pylist(),tbl['image'].to_pylist()):
        rgb=np.asarray(Image.open(io.BytesIO(im)).convert('RGB'))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        kp,ds=sift.detectAndCompute(gray,None)
        names[n]=(split,(int(Path(n).stem)-1)//100,hashlib.sha256(rgb.tobytes()).hexdigest())
        descriptors[n]=(np.float32([k.pt for k in kp]).reshape(-1,2),ds,gray)
pairs=set()
for n in scored:
    for m in names:
        if m!=n and names[m][1]==names[n][1]:
            pairs.add(tuple(sorted((n,m))))
def check(pair):
    n,m=pair
    xy1,d1,img1=descriptors[n]
    xy2,d2,img2=descriptors[m]
    if names[n][2]==names[m][2]:
        return (*pair,'exact',0,1.,1.,1.)
    if min(len(xy1),len(xy2))<12:
        return
    # Favor unique matches; a repeated crop texture alone is insufficient evidence.
    matches=cv2.BFMatcher().knnMatch(d1,d2,k=2)
    hits=[a for a,b in matches if a.distance<.73*b.distance]
    if len(hits)<12:
        return
    p=xy1[[m.queryIdx for m in hits]]
    q=xy2[[m.trainIdx for m in hits]]
    cv2.setRNGSeed(1191)
    aff,mask=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.)
    if aff is None or mask.sum()<12 or mask.mean()<.65:
        return
    scale=np.hypot(aff[0,0],aff[1,0])
    if not .85<=scale<=1.18:
        return
    inliers=p[mask.ravel().astype(bool)]
    if cv2.contourArea(cv2.convexHull(inliers))<.04*img1.size:
        return
    size=(img2.shape[1],img2.shape[0])
    warped=cv2.warpAffine(img1,aff,size)
    valid=cv2.warpAffine(np.ones_like(img1),aff,size,flags=cv2.INTER_NEAREST).astype(bool)
    if valid.mean()<.15:
        return
    corr=np.corrcoef(warped[valid],img2[valid])[0,1]
    if np.isfinite(corr) and corr>=.80:
        return (*pair,'registered',int(mask.sum()),float(mask.mean()),float(valid.mean()),float(corr))
with ThreadPoolExecutor(max_workers=4) as pool:
    hits=[h for h in pool.map(check,sorted(pairs)) if h]
fields=['image_a','image_b','type','inliers','inlier_fraction','overlap','gray_correlation','split_a','split_b']
with (ROOT/'画面重叠候选.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f)
    w.writerow(fields)
    for h in hits:
        w.writerow([*h,names[h[0]][0],names[h[1]][0]])
(ROOT/'画面重叠筛查.json').write_text(json.dumps(dict(scored=len(scored),train=sum(v[0]=='train' for v in names.values()),
                 val=sum(v[0]=='val' for v in names.values()),comparisons=len(pairs),detected=len(hits),
                 cv2=cv2.__version__,scope='all same scene train/val tiles versus each scored tile; no test images'),indent=2))
print(len(pairs),len(hits))
