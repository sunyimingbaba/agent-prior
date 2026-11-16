"""Shared serial tool rollout and exact generation-state/response-token contract."""
import hashlib
import re

import torch

ANSWER = re.compile(r'<answer>(.*?)</answer>', re.S)
TOOL = re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?')


class ToolBoundaryStop:
    def __init__(self,processor,prompt_tokens):
        self.processor,self.prompt_tokens=processor,prompt_tokens
    def __call__(self,input_ids,scores,**kwargs):
        text=self.processor.decode(input_ids[0,self.prompt_tokens:].tolist(),skip_special_tokens=True)
        call=TOOL.fullmatch(text.strip())
        return torch.tensor([bool(call and call.group(1)=='scene_classify')],device=input_ids.device)


def assistant_logprob(logits,target,eos_ids):
    # min_new_tokens=1 suppresses EOS at the first sampled token; match that policy.
    assert int(target[0]) not in eos_ids
    values=logits.float().clone()
    values[0,eos_ids]=-torch.inf
    return -torch.nn.functional.cross_entropy(values,target,reduction='sum')


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def messages_for(caption, system_prompt, few_shot=True):
    messages = [{'role': 'system', 'content': system_prompt}]
    if few_shot:
        messages += [
            {'role': 'user', 'content': [{'type': 'text', 'text': '查询文本: There is a harbor with several ships.'}]},
            {'role': 'assistant', 'content': '<tool_call>scene_classify</tool_call>'},
            {'role': 'user', 'content': '<result>harbor (0.61), port (0.24), ship (0.08)</result>'},
            {'role': 'assistant', 'content': '<answer>图中是一座港口，与查询文本相符。</answer>'}]
    messages.append({'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '查询文本: '+caption}]})
    return messages


def effective_response(ids, eos_ids, recorded_length):
    """Recorded generation length is authoritative, including EOS but excluding pad."""
    values = list(ids[:recorded_length])
    assert values and len(values) == recorded_length
    for i, token in enumerate(values):
        if token in eos_ids:
            return values[:i+1]
    return values


@torch.no_grad()
def rollout(model, processor, image, caption, tools, system_prompt, config,
            temperature=1., force_initial_tool=False):
    messages = messages_for(caption, system_prompt)
    events, rounds = [], []
    stop_reason = 'round_limit'
    if force_initial_tool:
        event = tools['scene_classify'](image)
        events.append({'name': 'scene_classify', **event, 'origin': 'software_contract_witness'})
        messages += [{'role': 'assistant', 'content': '<tool_call>scene_classify</tool_call>'},
                     {'role': 'user', 'content': '<result>'+event['text']+'</result>'}]
    hidden = None
    prior_mode = model.training
    model.eval()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    for round_index in range(config['max_rounds']):
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert rendered.endswith('<|im_start|>assistant\n')
        inputs = processor(text=[rendered], images=[image], return_tensors='pt').to('cuda')
        kwargs = dict(max_new_tokens=config['max_new_tokens_per_round'], min_new_tokens=1,
                      do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.,
                      top_k=0, top_p=1., repetition_penalty=1., use_cache=True,
                      output_hidden_states=True, return_dict_in_generate=True)
        output = model.generate(**inputs, **kwargs,
                                stopping_criteria=[ToolBoundaryStop(processor,inputs.input_ids.shape[1])])
        p = inputs.input_ids.shape[1]
        ids = output.sequences[0, p:].detach().cpu().tolist()
        assert len(ids) == len(output.hidden_states) > 0
        assert effective_response(ids, eos, len(ids)) == ids
        text = processor.decode(ids, skip_special_tokens=True)
        hidden = output.hidden_states[-1][-1][0, -1, :].detach().cpu().contiguous()
        assert list(hidden.shape) == [3584] and torch.isfinite(hidden).all()
        limited = len(ids) == config['max_new_tokens_per_round']
        rounds.append({'round': round_index+1, 'prompt_text': rendered,
            'prompt_text_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
            'message_roles': [x['role'] for x in messages], 'prompt_token_ids': inputs.input_ids[0].cpu().tolist(),
            'prompt_tokens': p, 'generated_ids': ids, 'generated_token_count': len(ids),
            'generated_text': text, 'token_limit': limited, 'eos_token_ids': eos,
            'state_input_token_id': int(output.sequences[0, p+len(ids)-2]),
            'last_emitted_token_id': ids[-1], 'state_tensor_sha256': tensor_sha(hidden),
            'generation_kwargs': {k: v for k, v in kwargs.items() if k not in ['output_hidden_states','return_dict_in_generate']}})
        del output, inputs
        call = TOOL.fullmatch(text.strip())
        if call and call.group(1) in tools and not limited:
            event = tools[call.group(1)](image)
            events.append({'name': call.group(1), **event, 'origin': 'model_requested_actual_tool',
                           'after_round': round_index+1})
            messages += [{'role': 'assistant', 'content': text},
                         {'role': 'user', 'content': '<result>'+event['text']+'</result>'}]
            continue
        answer = ANSWER.search(text)
        if limited:
            stop_reason = 'token_limit'
        elif '<result>' in text or '</result>' in text:
            stop_reason = 'model_generated_tool_result'
        elif TOOL.search(text):
            stop_reason = 'mixed_or_unknown_tool_output'
        elif answer and answer.group(1).strip():
            stop_reason = 'answer'
        else:
            stop_reason = 'format_failure' if text.strip() else 'empty'
        break
    model.train(prior_mode)
    assert hidden is not None
    usable = stop_reason == 'answer' and all(not x['token_limit'] for x in rounds) \
        and all('addCriterion' not in x['generated_text'] for x in rounds)
    trace = {'rounds': rounds, 'tool_events': events, 'tools_used': len(events),
             'quality_usable': usable, 'stop_reason': stop_reason,
             'trajectory_text': '\n'.join(x['generated_text'] for x in rounds),
             'state_tensor_sha256': tensor_sha(hidden),
             'state_rule': 'last actual generate, last valid decode step, final layer, final position',
             'forced_tool_witness': force_initial_tool}
    return hidden, trace


def response_logprob(model, processor, image, record):
    """Recompute assistant probabilities only; right padding is trimmed before forward."""
    ids = effective_response(record['generated_ids'], record['eos_token_ids'], record['generated_token_count'])
    inputs = processor(text=[record['prompt_text']], images=[image], return_tensors='pt').to('cuda')
    assert inputs.input_ids[0].tolist() == record['prompt_token_ids']
    p = inputs.input_ids.shape[1]
    target = torch.tensor(ids, device='cuda', dtype=torch.long)
    full = torch.cat([inputs.input_ids, target[None]], dim=1)
    inputs['input_ids'] = full
    inputs['attention_mask'] = torch.ones_like(full)
    outputs = model(**inputs, use_cache=False, return_dict=True)
    logits = outputs.logits[0, p-1:p+len(ids)-1].float()
    assert logits.shape[0] == len(ids)
    value = assistant_logprob(logits,target,record['eos_token_ids'])
    return value, {'effective_tokens': len(ids), 'post_generation_padding_ignored': len(record['generated_ids'])-len(ids),
                   'forward_input_tokens': full.shape[1], 'response_loss_tokens': len(ids)}
