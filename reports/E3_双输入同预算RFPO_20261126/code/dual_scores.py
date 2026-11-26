import numpy as np
import torch
from functools import lru_cache
import sys
from study import STAGE, PREV, FORM, RFPO, ARMS, CONTRASTS, read, dump, load, sha, feature_path, banks, bank_for, digest
from study import calculate, summary_by_seed, paired, fixed, rr

@lru_cache(maxsize=2)
def saved_files(folder):
    return {key:val for j in range(2) for key,val in read(folder/('complete_'+str(j)+'.json'))['files'].items()}


def features(tag, arm, seed, rows, reference=False):
    if arm == 'retained' and reference:
        folder=STAGE/'reference/evaluate'/(seed+'_retained')
        path=folder/(tag+'.pt')
        assert sha(path)==read(folder/tag/'complete.json')['features']
        result=load(path)
        assert result['adapter']==read(PREV/'strict_rfpo/inputs/lock.json')['starts'][seed]['R2']
    elif arm == 'retained':
        if tag in ('primary','official'):
            result = load(RFPO/'vectors'/(seed+'_deployed_'+tag+'.pt'))
        else:
            view = tag.split('_')[1]
            baseline = read(PREV/'strict_rfpo/inputs/lock.json')['starts'][seed]['R2']
            vectors = []
            for row in rows:
                path = feature_path(row,seed,view)
                pack = load(path)
                assert pack['rgb'] == row['rgb_sha256']
                if row['tag']=='saved' and view=='scene':
                    assert sha(path) == saved_files(path.parent)[row['image']]
                else:
                    assert pack['adapter'] == baseline
                vectors.append(pack['visual'][-1])
            return torch.stack(vectors)
    else:
        folder = STAGE/'evaluate'/(seed+'_'+arm)
        path = folder/(tag+'.pt')
        assert sha(path) == read(folder/tag/'complete.json')['features']
        result = load(path)
        assert result['adapter'] == read(STAGE/'models'/(seed+'_'+arm)/'complete.json')['final']
    assert result['images'] == [r['image'] for r in rows]
    return result['features']


def panel_outcomes(rows, vectors, text, masks):
    values = []
    for row, visual in zip(rows, vectors):
        group = int(row['tag'] in ('extra','saved'))
        native, hard, maps = text[group]
        ix = maps[row['image']]
        normal_scores = native @ visual.double().numpy()
        det_scores = hard @ visual.double().numpy()
        pos = ix['native']
        negatives = masks[group].copy()
        negatives[pos] = False
        alts = sorted(set(pos)-{pos[0]})
        ranks = [float(rr(normal_scores[i],normal_scores[negatives])) for i in alts]
        correct = [int(det_scores[ix[k][0]] > det_scores[ix[k][1]]) for k in 'CDE']
        values.append((correct,ranks))
    return values


def gallery_outcomes(rows, vectors, indices, native, details):
    sim = fixed(vectors,native['native'])
    hard = fixed(vectors,details['hard'])
    order = native['caption_order'].numpy().reshape(210,5)
    values = []
    for j,(row,idx) in enumerate(zip(rows,indices)):
        pos = order[idx];mask = np.ones(sim.shape[1], dtype=bool);mask[pos] = False
        ix = details['maps']['primary'][row['image']]
        correct = [int(hard[j,ix[k][0]] > hard[j,ix[k][1]]) for k in 'CDE']
        ranks = [float(rr(sim[j,col],sim[j,mask])) for col in sorted(set(pos)-{pos[0]})]
        values.append((correct,ranks))
    return values


def summarize(data, labels):
    mix = {s: {k: data['dualjoint'][s][k]-data['singlejoint'][s][k]-data['dualplain'][s][k]+data['singleplain'][s][k]
               for k in data['dualjoint'][s]} for s in 'ab'}
    return dict(levels={a:summary_by_seed(v,labels) for a,v in data.items()},
                contrasts={a+'-'+b:paired(data[a],data[b],labels) for a,b in CONTRASTS},
                interaction=summary_by_seed(mix,labels))


def describe(raw, rows, mask=None):
    idx = list(range(len(rows))) if mask is None else np.flatnonzero(mask).tolist()
    labels = [rows[i]['pair'] for i in idx]
    data = {}
    for arm in ARMS:
        data[arm] = {}
        for seed in 'ab':
            vals = [raw[arm][seed][i] for i in idx]
            data[arm][seed] = {k:np.array([np.mean(x[j]) for x in vals]) for j,k in enumerate(('CDE','mrr'))}
    result = summarize(data,labels)
    result.update(images=len(idx),pairs=len(set(labels)),joint={})
    for a,b in CONTRASTS:
        diffs = {}
        for seed in 'ab':
            win,loss = [],[]
            for i in idx:
                x,p = raw[a][seed][i]; y,q = raw[b][seed][i]
                dp = np.asarray(p)-q
                win.append(float(np.mean(dp>0)) if sum(x)>sum(y) else 0.)
                loss.append(float(np.mean(dp<0)) if sum(x)<sum(y) else 0.)
            diffs[seed] = dict(win=np.array(win),loss=np.array(loss),net=np.subtract(win,loss))
        result['joint'][a+'-'+b] = summary_by_seed(diffs,labels)
    return result


def main(reference=False):
    torch.set_num_threads(1)
    packs = banks()
    text = [(p['native'].double().numpy(),p['hard'].double().numpy(),p['maps']) for p in packs]
    masks = [np.array([int(digest(','.join(map(str,t)))[-1],16)%2==0 for t in p['native_tokens'].tolist()]) for p in packs]
    native = load(FORM/'results/ucm_text.pt'); details = load(RFPO/'records/text.pt')
    prim = read(RFPO/'inputs/panel.json'); official = read(FORM/'inputs/ucm.json')
    offmap = {r['rgb_sha256']:i for i,r in enumerate(official)}
    link = [r for r in prim if r['rgb_sha256'] in offmap]; assert len(link)==106
    link_off = [offmap[r['rgb_sha256']] for r in link]
    link_prim = [j for j,r in enumerate(prim) if r['rgb_sha256'] in offmap]
    held = read(STAGE/'inputs/holdout.json'); repeat = read(STAGE/'inputs/repeat.json')
    flags = {r['image']:r['group'] for r in read(STAGE/'inputs/caption_cues.json')}
    result = {}; logs = []; full = []
    for tag,rows in [('hold_scene',held),('hold_first',held),('repeat_scene',repeat),('repeat_first',repeat),
                     ('primary',prim),('official',official)]:
        formal = tag in ('primary','official')
        raw = {a:{} for a in ARMS}; metrics = {a:{} for a in ARMS}
        for arm in ARMS:
            for seed in 'ab':
                v = features(tag,arm,seed,rows,reference)
                if formal:
                    metrics[arm][seed] = calculate(tag,v,rows)
                    full.extend(dict(tag=tag,arm=arm,seed=seed,image=row['image'],
                        values={k:float(x[i]) for k,x in metrics[arm][seed].items()}) for i,row in enumerate(rows))
                    vs = v[link_off if tag=='official' else link_prim]
                    vals = gallery_outcomes(link,vs,link_off,native,details)
                else:
                    vals = panel_outcomes(rows,v,text,masks)
                raw[arm][seed] = vals
                logs.extend(dict(panel=tag,arm=arm,seed=seed,image=r['image'],CDE=x,mrr_by_caption=p)
                            for r,(x,p) in zip(link if formal else rows,vals))
        if formal:
            result[tag] = summarize(metrics,[r.get('pair',r['image']) for r in rows])
            result['shared_'+('first' if tag=='official' else 'scene')] = describe(raw,link)
        else:
            result[tag] = describe(raw,rows)
            if tag.startswith('hold'):
                result[tag+'_unmentioned'] = describe(raw,rows,[flags[r['image']]=='unmentioned' for r in rows])
    decision = {}
    for comp in ('dualjoint-singlejoint','dualjoint-dualplain','dualjoint-retained'):
        decision[comp] = all(result[t]['contrasts'][comp][k]['mean']>=0 for t in ('hold_first','hold_scene') for k in ('CDE','mrr')) and all(result[t]['joint'][comp]['net']['mean']>0 for t in ('hold_first','hold_scene'))
    cp = [result[t]['contrasts']['dualjoint-retained'][k] for t,k in [('primary','CDE'),('official','mR')]]
    result['decision'] = dict(cross_input=decision,promote=all(min(x['seeds'])>0 and x['interval'][0]>0 for x in cp))
    dest=STAGE/'reference' if reference else STAGE/'results'
    dump(dest/'ucm.json',result)
    dump(dest/'caption_outcomes.json',[r for r in logs if r['arm']=='retained'] if reference else logs)
    dump(dest/'formal_images.json',[r for r in full if r['arm']=='retained'] if reference else full)


if __name__ == '__main__':
    main(sys.argv[1:]==['reference'])
