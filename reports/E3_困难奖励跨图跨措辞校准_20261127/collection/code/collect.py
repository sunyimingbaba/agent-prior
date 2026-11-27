"""Recover predictor states at the last content token and at the archived EOS."""
from pathlib import Path
import importlib.util
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
PRIOR=BASE/'joint_answer_pairs_20261127'
OLD=PRIOR/'controlled98'
spec=importlib.util.spec_from_file_location('previous_control',OLD/'code/collect.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read,sha,dump=old.read,old.sha,old.dump

def tokenize(proc, stmt, eos):
    text='<answer>'+stmt+'</answer>'
    tok=proc.tokenizer(text,return_offsets_mapping=True,add_special_tokens=False)
    ids=list(tok['input_ids'])+[eos]
    bounds=list(map(tuple,tok['offset_mapping']))
    start=len('<answer>')
    end=start+len(stmt)
    word_end=max(i for i,(lo,hi) in enumerate(bounds)
                 if any(text[c].isalnum() for c in range(max(lo,start),min(hi,end))))
    lo,hi=bounds[word_end]
    assert lo>=start and hi<=end and any(c.isalnum() for c in text[lo:hi])
    assert proc.tokenizer.decode(ids[:-1],skip_special_tokens=False)==text
    return text,ids,word_end,list(bounds[word_end])

def force(model,inp,paths,content_indices,eos):
    n=inp['input_ids'].shape[1]
    result=model.generate(**inp,max_new_tokens=64,do_sample=True,temperature=1.,
            logits_processor=old.LogitsProcessorList([old.FixedTokens(paths,n,eos)]),
            output_scores=True,output_hidden_states=True,return_dict_in_generate=True)
    states=[]; content=[]
    for j,path in enumerate(paths):
        assert old.cut(result.sequences[j,n:].tolist(),model.generation_config.eos_token_id)==path
        states.append(result.hidden_states[len(path)-1][-1][j,-1].detach().cpu().clone())
        content.append(result.hidden_states[content_indices[j]+1][-1][j,-1].detach().cpu().clone())
    return torch.stack(states),torch.stack(content)

@torch.inference_mode()
def main():
    old.setup();torch.set_num_threads(1)
    model,proc=old.policy('R0'); table=old.table('train')
    model_sha=sha(BASE/'second_answer_5090_20261124/R0/adapter/adapter_model.safetensors')
    eos=proc.tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert eos==151645
    catalog=read(ROOT/'inputs/catalog.json'); inventory=read(ROOT/'inputs/inventory.json')
    assert sha(ROOT/'inputs/catalog.json')==inventory['catalog']
    for panel in ('train98','pool270'):
        source=PRIOR/'fresh98' if panel=='train98' else BASE/'aftertool_joint_pool_20261127'
        rows=read(source/('inputs/rows.json' if panel=='train98' else 'scoring/inputs/rows.json'))
        rawdir=source/('raw/c' if panel=='train98' else 'sampling/raw/a')
        files={}; folder=ROOT/panel/'raw';folder.mkdir(exist_ok=True,parents=True)
        expected=98 if panel=='train98' else 238
        for row in rows:
            name=row['image'];r=catalog[panel][name]
            if r['attribute'] not in inventory['common_attributes'] or r['pair'] not in inventory['panels'][panel]['complete_pairs']:
                continue
            target=folder/(name+'.pt')
            previous=rawdir/(name+'.pt')
            assert sha(previous)==r['source'] and r['rgb']==row['rgb_sha256']
            paths0=('A',) if panel=='train98' else ('D','E')
            texts=[]; paths=[]; indices=[]; offsets=[]
            for k in paths0:
                for q in row[k]:
                    text,ids,idx,bound=tokenize(proc,q,eos)
                    texts.append(text);paths.append(ids);indices.append(idx);offsets.append(bound)
            if not target.exists():
                raw=torch.load(previous,weights_only=True,map_location='cpu')
                assert raw['adapter']==model_sha and raw['eligible'] and raw['query']==row['orig_query']
                assert raw['context']['answer']==row['feedback']
                inp=old.prepared(proc,old.rgb(table,row),row,raw)
                torch.manual_seed(raw['rng'])
                n=len(paths); repeat=8//n
                states,content=force(model,inp,paths*repeat,indices*repeat,eos)
                assert all(torch.equal(states[i],states[i%n]) and torch.equal(content[i],content[i%n]) for i in range(8))
                if panel=='train98':
                    older=OLD/'raw'/(name+'.pt')
                    assert sha(older)==r['controlled_sha']
                    ref=torch.load(older,weights_only=True,map_location='cpu')
                    assert ref['ids']==paths and torch.equal(ref['states'],states[:n])
                old.savept(target,dict(image=name,rgb=r['rgb'],pair=r['pair'],attribute=r['attribute'],side=r['side'],
                    source=r['source'],row=row['row'],eos=eos,texts=texts,ids=paths,
                    content_indices=indices,content_offsets=offsets,states=states[:n].clone(),
                    content_states=content[:n].clone(),adapter=model_sha,prefix=raw['context']['prefix'],code=sha(Path(__file__))))
            current=torch.load(target,map_location='cpu',weights_only=True)
            assert current['source']==r['source'] and current['ids']==paths and current['content_indices']==indices
            assert current['code']==sha(Path(__file__)) and torch.isfinite(current['states']).all() and torch.isfinite(current['content_states']).all()
            files[name]=sha(target);print(panel,len(files),name,flush=True)
        assert len(files)==expected
        dump(ROOT/panel/'generation.json',dict(images=expected,files=files,adapter=model_sha,
             code=sha(Path(__file__)),source=sha(OLD/'code/collect.py'),catalog=sha(ROOT/'inputs/catalog.json'),
             gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)))
if __name__=='__main__':main()
