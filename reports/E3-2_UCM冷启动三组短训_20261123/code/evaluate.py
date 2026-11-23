"""Generate exactly one greedy deployment prior per validation image with the real scene tool."""
import sys
import torch
from shared import ROOT, REBUILD, ASSETS, read, dump, sha, setup, table, rgb
from readouts import load_model, clip
from policy import policy
from rollout import rollout_one, SYSTEM_PROMPT
from tools import scene_classify_with_conf

@torch.inference_mode()
def run(arm):
    setup()
    inputs = REBUILD/'paired_views_20261120/inputs'
    init = read(ROOT/'initial.json')
    weight = sha(ROOT/arm/'adapter/adapter_model.safetensors')
    source = init['adapter'] if arm=='R0' else read(ROOT/arm/'complete.json')['final']
    assert source == weight and sha(inputs/'validation.json')==init['val']
    lock = read(inputs/'lock.json')
    assert sha(inputs/'retrieval_text.pt') == lock['retrieval_text.pt']
    assert sha(inputs/'e1_extension.json') == lock['e1_extension.json']
    for name, expected in read(inputs/'e1_extension.json')['files'].items():
        assert sha(inputs/'e1'/name)==expected
    model, proc = policy(arm)
    reader = load_model(False)
    tool,pre = clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    tool = tool.float().eval().cuda()
    cp = torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head = torch.nn.Linear(cp['feat_dim'],len(cp['classes']))
    head.load_state_dict(cp['head'])
    head = head.eval().cuda()
    val, t = read(inputs/'validation.json'), table('val')
    texts = torch.load(inputs/'retrieval_text.pt',map_location='cpu',weights_only=True)
    assert len(val)==len(t)==210 and [x['row'] for x in val]==list(range(210))
    raw = clip.tokenize([q for caps in t['caption'].to_pylist() for q in caps], context_length=77-reader.prompt_length)
    assert torch.equal(texts['native_tokens'][texts['caption_order']],raw)
    assert (reader.txt_embeds(raw[:4].cuda())['text_feats'].cpu()-texts['native'][texts['caption_order'][:4]]).abs().max()<2e-5
    out = ROOT/arm/'eval'
    out.mkdir(exist_ok=True)
    generator = model.generate
    def record(**kw):
        response = generator(**kw)
        ids = response.sequences[0,kw['input_ids'].shape[1]:].tolist()
        rounds.append(dict(length=len(ids),eos=ids[-1] in model.generation_config.eos_token_id,
                           ids=ids,text=proc.decode(ids,skip_special_tokens=True)))
        return response
    model.generate = record
    rows = []
    for case in val:
        path = out/f"{case['row']:03d}.pt"
        if path.exists():
            entry = torch.load(path,map_location='cpu',weights_only=True)
            assert entry['adapter']==weight and entry['image']==case['image']
        else:
            im = rgb(t,case)
            rounds,events = [],[]
            def scene(image):
                if not events:
                    answer, conf = scene_classify_with_conf(image,model=tool,preprocess=pre,cls_head=head,cls_names=list(cp['classes']))
                    events.append((answer,float(conf)))
                else:
                    events.append(events[0])
                return events[-1][0]
            h,used,filtered = rollout_one(model,proc,im,case['orig_query'],{'scene_classify':scene},3,64,
                                         'cuda',temperature=0.,system_prompt=SYSTEM_PROMPT)
            assert h.shape==(3584,) and h.dtype==torch.bfloat16 and torch.isfinite(h).all() and used==len(events) and filtered==0
            v = reader.img_embeds(reader.preprocess(im)[None].cuda(),agent_hidden=h[None].cuda())['image_feats'][0].cpu()
            assert torch.isfinite(v).all() and abs(v.norm().item()-1)<1e-4
            entry = dict(image=case['image'],row=case['row'],rgb=case['rgb_sha256'],adapter=weight,query=case['orig_query'],
                         rounds=rounds,tools=events,prior=h,feature=v)
            tmp = path.with_suffix('.part')
            torch.save(entry,tmp)
            tmp.replace(path)
        rows.append(sha(path))
        if len(rows)%10==0:
            print(arm,len(rows),flush=True)
    dump(out/'complete.json', dict(adapter=weight,rows=rows,gallery=lock['retrieval_text.pt'],
         reader=sha(ASSETS/'ucm_main_domain_tool.ckpt'),tool=sha(ASSETS/'ucm_scene_cls.pt')))

if __name__=='__main__':
    run(sys.argv[1])
