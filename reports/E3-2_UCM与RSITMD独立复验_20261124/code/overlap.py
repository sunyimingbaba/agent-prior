"""Sensitivity screen for overlapping crops using local features and a similarity transform."""
import io,sys
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import pyarrow as pa
from PIL import Image
from shared import ROOT,dump,read
SETS=('ucm_train','ucm_val','ucm_test','rsitmd_test')

def main(base):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(20261124)
    sift=cv2.SIFT_create(nfeatures=160)
    desc=[]; points=[]; names=[]; parts=[]
    for part in SETS:
        dn,split=part.split('_')
        t=pa.ipc.open_file(pa.memory_map(str(base/dn/f'{dn}_caption_karpathy_label_{split}.arrow'))).read_all()
        for j,img in enumerate(t['image'].to_pylist()):
            gray=np.asarray(Image.open(io.BytesIO(img)).convert('L').resize((192,192),Image.Resampling.LANCZOS))
            k,v=sift.detectAndCompute(gray,None)
            desc.append(v if v is not None else np.empty((0,128),np.float32))
            points.append(np.asarray([z.pt for z in k],np.float32))
            names.append(t['path'][j].as_py());parts.append(part)
    # Candidate search pools references jointly but removes descriptors from the query itself.
    db=np.concatenate(desc); owner=np.repeat(np.arange(len(names)),[len(v) for v in desc])
    index=cv2.flann_Index(db,dict(algorithm=1,trees=4))
    matcher=cv2.BFMatcher()
    edges=[]; omitted={}
    for i,part in enumerate(parts):
        if part not in SETS[2:] or len(desc[i])<12:continue
        nbr,d2=index.knnSearch(desc[i],5,params=dict(checks=40))
        votes=Counter()
        for ix,ds in zip(nbr,d2):
            keep=owner[ix]!=i
            if keep.any():
                j=int(owner[ix[keep][0]])
                if j<i:votes[j]+=1
        for j,n in votes.most_common(24):
            if n<3 or len(desc[j])<12:continue
            matches=[a for a,b in matcher.knnMatch(desc[i],desc[j],k=2) if a.distance < .72*b.distance]
            # Enforce unique target locations, including SIFT's duplicate orientations.
            unique={}
            for m in sorted(matches,key=lambda m:m.distance):
                k=tuple(np.rint(points[j][m.trainIdx]).astype(int))
                if k not in unique:unique[k]=m
            matches=list(unique.values())
            if len(matches)<12:continue
            a=np.float32([points[i][m.queryIdx] for m in matches]);b=np.float32([points[j][m.trainIdx] for m in matches])
            tf,mask=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.0,maxIters=3000)
            if tf is None:continue
            mask=mask.ravel().astype(bool)
            if mask.sum()<12 or mask.mean()<.6:continue
            area=min(cv2.contourArea(cv2.convexHull(z[mask]))/(192*192) for z in (a,b))
            if area<.10:continue
            edges.append(dict(query_set=part,query=names[i],reference_set=parts[j],reference=names[j],inliers=int(mask.sum()),
                              candidate_matches=len(matches),coverage=round(area,4),transform=tf.tolist()))
            if parts[j]!=part or (parts[j],names[j]) not in omitted:
                omitted[part,names[i]]=True
    dump(ROOT/'inputs/overlap_screen.json',dict(edges=edges,exclude={dn:[name for p,name in omitted if p==dn+'_test'] for dn in ('ucm','rsitmd')},
        method='SIFT at 192px, top 24 candidates, 2px RANSAC similarity, >=12 unique destination points, >=60% inliers and >=10% hull coverage in both views. Sensitivity analysis only.'))
if __name__=='__main__':main(Path(sys.argv[1]))
