import argparse,csv,cv2,numpy as np,json
from pathlib import Path
from itertools import combinations
cv2.setNumThreads(1)
parser=argparse.ArgumentParser(description='Find image pairs for visual overlap review; outputs candidates, not final judgments.')
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--images-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
p=args.manifest;rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
root=args.images_dir;sift=cv2.SIFT_create(nfeatures=1200,contrastThreshold=.012);bf=cv2.BFMatcher();data=[]
for r in rows:
 img=cv2.imread(str(root/r['图像编号']),0);kp,des=sift.detectAndCompute(img,None);data.append((r,kp,des))
out=[]
for (r1,k1,d1),(r2,k2,d2) in combinations(data,2):
 if d1 is None or d2 is None:continue
 matches=bf.knnMatch(d1,d2,k=2);raw=[a for p in matches if len(p)==2 for a,b in [p] if a.distance<0.77*b.distance]
 raw.sort(key=lambda m:m.distance);good=[];usedq=set();usedt=set()
 for m in raw:
  q=tuple(round(v/3) for v in k1[m.queryIdx].pt);t=tuple(round(v/3) for v in k2[m.trainIdx].pt)
  if q in usedq or t in usedt:continue
  usedq.add(q);usedt.add(t);good.append(m)
 if len(good)<5:continue
 a=np.float32([k1[m.queryIdx].pt for m in good]);b=np.float32([k2[m.trainIdx].pt for m in good]);M,mask=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=4000,confidence=.995)
 if M is None:continue
 sel=mask.ravel().astype(bool);num=int(sel.sum())
 if num<5:continue
 aa=a[sel];bb=b[sel];ar1=cv2.contourArea(cv2.convexHull(aa))/(256*256);ar2=cv2.contourArea(cv2.convexHull(bb))/(256*256)
 scale=float(np.linalg.norm(M[:,0]));ang=float(np.degrees(np.arctan2(M[1,0],M[0,0])))
 if min(ar1,ar2)<.005 or not .4<scale<2.5:continue
 out.append({'a':r1['图像编号'],'b':r2['图像编号'],'inliers':num,'candidates':len(good),'area1':round(ar1,3),'area2':round(ar2,3),'scale':round(scale,2),'angle':round(ang,1)})
out.sort(key=lambda a:(a['inliers'],min(a['area1'],a['area2'])),reverse=True)
args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('flags',len(out))
for x in out[:30]:print(x)
