"""Freeze opposite visual pairs from case-independent inspection, then identify overlapping aerial tiles."""
import hashlib, json
from pathlib import Path
import numpy as np
import cv2
from sheets import candidates,picture,OUT,REP,read
# In each pair the first image supports the first sentence in each fixed wording.
PAIRS={
'field_axis':[(56,25),(58,69)], 'field_color':[(28,50),(53,2)],
'surf':[(335,307),(345,332),(341,312),(348,340)],
'freeway_axis':[(817,821),(859,830),(851,863)],'freeway_color':[(805,868),(804,841)],
'bridge_axis':[(1407,1422),(1435,1449)],'bridge_slope':[(1402,1416)],
'river_bank':[(1637,1658),(1644,1664)],'river_axis':[(1602,1639)],
'berths':[(1008,1024),(1045,1036),(1010,1026)],'harbor_rows':[(1070,1005)],
'parking_fill':[(1517,1562),(1558,1567)],'parking_axis':[(1571,1580)],
'runway_bars':[(1707,1712),(1728,1730),(1744,1753),(1777,1765)],
'house_pool':[(1829,1851),(1880,1827)],'house_light':[(1811,1804),(1833,1879),(1877,1860)],
'court_pool':[(2004,2013),(2051,2050),(2074,2067)],'crosswalk':[(1158,1162),(1160,1109)],
'forest_color':[(735,727),(717,730),(762,744),(749,733)],
'golf_sand':[(904,908),(925,922),(938,932),(961,960),(973,977),(910,914),(928,939),(950,949)],
'tank_number':[(1908,1917),(1925,1923),(1926,1928),(1958,1955),(1968,1961),(1975,1980)],
'infield_tone':[(272,268),(275,274),(211,262)],
'court_axis':[(2009,2005),(2064,2012)],'street_axis':[(611,607),(651,641)],
'road_diagonal':[(1102,1112),(1126,1141),(1133,1144),(1134,1153)],
'chaparral_tone':[(563,509),(566,521),(561,552)]}
NEW={
'forest_color':('forest', [
('Green foliage dominates the woodland.','Brown foliage dominates the woodland.'),
('The forest canopy looks green.','The forest canopy looks brown.'),
('The cover of leaves in this forest is mostly green.','The cover of leaves in this forest is mostly brown.'),
('The treetops have a greenish cast.','The treetops have a brownish cast.'),
('Green is the predominant color of the trees.','Brown is the predominant color of the trees.')]),
'golf_sand':('golf course',[
('A pale sand bunker is visible on the golf course.','There is no pale sand bunker visible on the golf course.'),
('The golf course includes a light sand trap.','The golf course shows no light sand trap.'),
('At least one bright sandy hazard is visible on the grass.','The grassy course has no visible bright sandy hazard.'),
('A light patch of bunker sand stands out from the fairway.','No light patch of bunker sand stands out from the fairway.'),
('One can see a pale sand trap on this course.','No pale sand trap can be seen on this course.')]),
'tank_number':('storage tanks',[
('There is one visible circular tank.','There are multiple visible circular tanks.'),
('A single round tank is visible.','Several round tanks are visible.'),
('Only one circular tank appears in this view.','More than one circular tank appears in this view.'),
('The image contains one round storage vessel.','The image contains two or more round storage vessels.'),
('One tank with a circular footprint is in sight.','Several tanks with circular footprints are in sight.')]),
'infield_tone':('baseball diamond',[
('The baseball infield is pale beige.','The baseball infield is reddish brown.'),
('The infield has a light tan color.','The infield has a red-brown color.'),
('The dirt inside the diamond is light beige.','The dirt inside the diamond is reddish brown.'),
('The enclosed playing area has a pale tan surface.','The enclosed playing area has a rust-colored surface.'),
('A light, sandy color fills the infield.','A reddish earthy color fills the infield.')]),
'court_axis':('tennis courts',[
('The tennis courts run from left to right.','The tennis courts run from top to bottom.'),
('The long axes of the courts are horizontal.','The long axes of the courts are vertical.'),
('The court sidelines lie across the image width.','The court sidelines lie along the image height.'),
('The long sides of the playing rectangles extend sideways.','The long sides of the playing rectangles extend vertically.'),
('Play is oriented across the frame.','Play is oriented up and down the frame.')]),
'road_diagonal':('intersection',[
('The crossing roads run diagonally across the picture.','The crossing roads run horizontally and vertically in the picture.'),
('The intersection is rotated relative to the image edges.','The intersection is aligned with the image edges.'),
('The streets form a tilted cross in the frame.','The streets form an upright cross in the frame.'),
('Both streets are oblique to the borders of the picture.','The streets run parallel to the borders of the picture.'),
('The crossing is set at a diagonal.','The crossing follows the horizontal and vertical axes.')]),
'chaparral_tone':('chaparral',[
('The scrubland has a gray-green background.','The scrubland has a sandy brown background.'),
('The ground among the shrubs looks grayish green.','The ground among the shrubs looks tan.'),
('Gray-green tones dominate the spaces between bushes.','Sandy brown tones dominate the spaces between bushes.'),
('The scrub has a cool gray-green hue.','The scrub has a warm brown hue.'),
('The background of the shrubland appears grayish green.','The background of the shrubland appears sandy brown.')])}
# Second visual pass over juxtaposed pairs, before generation or scoring.
# Eliminate ambiguous orientations, tones, objects or shoreline descriptions.
PAIRS.update(field_axis=[],field_color=[],surf=[(345,332),(348,340)],bridge_slope=[],
             river_bank=[],berths=[],house_pool=[(1829,1851)],house_light=[],court_axis=[],
             court_pool=[(2004,2013),(2051,2050)],crosswalk=[(1158,1162)],
             parking_fill=[(1517,1562)],forest_color=[(735,727),(717,730)],
             tank_number=[(1908,1917),(1925,1923),(1926,1928),(1958,1955),(1975,1980)],
             road_diagonal=[(1102,1112),(1126,1141),(1133,1144)])
def main():
    t,idx,allowed=candidates();ref={}
    for p in REP.glob('E3*/**/inputs/panel.json'):
        rs=read(p)
        if isinstance(rs,list):
            for r in rs:
                if r.get('side')==0 and all(k in r for k in 'ABCDE'):ref.setdefault(r['attribute'],r)
    cv2.setNumThreads(1);cv2.setRNGSeed(20261125);sift=cv2.SIFT_create(nfeatures=200);matcher=cv2.BFMatcher()
    cache={}
    def get(n):
        if n not in cache:
            im=picture(t,idx[n]);gray=np.asarray(im.convert('L').resize((192,192),3))
            k,ds=sift.detectAndCompute(gray,None)
            cache[n]=(hashlib.sha256(im.tobytes()).hexdigest(),np.float32([v.pt for v in k]),ds)
        return cache[n]
    def near(a,b):
        ha,x,da=get(a);hb,y,db=get(b)
        if ha==hb:return True
        if da is None or db is None or min(len(da),len(db))<12:return False
        ms=[u for u,v in matcher.knnMatch(da,db,k=2) if u.distance<.72*v.distance]
        ix={tuple(y[z.trainIdx].round().astype(int)):z for z in sorted(ms,key=lambda m:-m.distance)}
        if len(ix)<12:return False
        p=np.float32([x[z.queryIdx] for z in ix.values()]);q=np.float32([y[z.trainIdx] for z in ix.values()])
        tf,mask=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=3000)
        if tf is None:return False
        hit=mask.ravel().astype(bool)
        return hit.sum()>=12 and hit.mean()>=.6 and min(cv2.contourArea(cv2.convexHull(v[hit])) for v in (p,q))>=.1*192**2
    seen=sorted(set(idx)-set(allowed));rows=[];removed=[];seen_ids=set()
    prompts={}
    for row in ref.values():prompts[row['scene_id']]=row['scene_query']
    for attr,pairs in PAIRS.items():
        if not pairs:continue
        if attr in NEW:scene,words=NEW[attr]
        else:r=ref[attr];scene=r['scene'];words=[r[k] for k in 'ABCDE']
        for a,b in pairs:
            names=[f'{n}.tif' for n in (a,b)];sid=(a-1)//100
            assert sid==(b-1)//100 and all(n in allowed and n not in seen_ids for n in names),names
            seen_ids.update(names)
            # Restrict pair matching to this land-use category; also check exact RGB across all categories below.
            other=[n for n in seen if (int(n[:-4])-1)//100==sid]
            hit=next(((n,s) for n in names for s in other if near(n,s)),None)
            if hit or near(*names):removed.append([names,hit or names]);continue
            seen+=names
            for i,n in enumerate(names):
                j=idx[n];caps=t['caption'][j].as_py();assert t['split'][j].as_py()=='train'
                rows.append(dict(image=n,pair=f'{a}-{b}',rgb_sha256=get(n)[0],side=i,row=j,split='train',scene_id=sid,scene=scene,attribute=attr,
                    scene_query=prompts.get(sid,'Scene: '+({'forest':'a forest','golf course':'a golf course','storage tanks':'storage tanks','chaparral':'chaparral'}.get(scene,scene))+'.'),
                    orig_query=caps[0],captions=caps,**{k:[v[i],v[1-i]] for k,v in zip('ABCDE',words)}))
    banned={r['rgb_sha256'] for p in REP.glob('E3*/**/inputs/panel.json') for r in (read(p) if isinstance(read(p),list) else []) if 'rgb_sha256' in r}
    assert len(rows)==len({r['rgb_sha256'] for r in rows}) and banned.isdisjoint(r['rgb_sha256'] for r in rows)
    for f,z in [('fresh',rows),('fresh_excluded',removed)]: (OUT/'inputs'/(f+'.json')).write_text(json.dumps(z,indent=2,ensure_ascii=False)+'\n')
    print(len(rows),len(removed))
if __name__=='__main__':main()
