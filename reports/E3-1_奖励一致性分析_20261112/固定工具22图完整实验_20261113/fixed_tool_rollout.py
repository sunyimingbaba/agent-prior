"""Controlled answer sampling after one supplied, actually executed scene tool."""
import hashlib
import re
import torch

ANSWER = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
CALL = re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?')


def initial_messages(caption, system_prompt):
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': [{'type': 'text', 'text': '查询文本: There is a harbor with several ships.'}]},
        {'role': 'assistant', 'content': '<tool_call>scene_classify</tool_call>'},
        {'role': 'user', 'content': '<result>harbor (0.61), port (0.24), ship (0.08)</result>'},
        {'role': 'assistant', 'content': '<answer>图中是一座港口，与查询文本相符。</answer>'},
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '查询文本: '+caption}]}]


@torch.no_grad()
def rollout_fixed_tool(model, processor, image, caption, tools, max_rounds, max_new_tokens,
                       device, verbose=False, injected_h_end=False, system_prompt=None, temperature=0.0):
    assert system_prompt and not injected_h_end and max_rounds == 1
    result = tools['scene_classify'](image)
    assert result is not None, 'Fixed tool context requires an accepted actual tool result.'
    fixed_call = {'role': 'assistant', 'content': '<tool_call>scene_classify</tool_call>'}
    fixed_result = {'role': 'user', 'content': '<result>'+result+'</result>'}
    messages = initial_messages(caption, system_prompt)+[fixed_call, fixed_result]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1,
        do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.0,
        output_hidden_states=True, return_dict_in_generate=True)
    new_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
    new_text = processor.decode(new_ids, skip_special_tokens=True)
    hidden = outputs.hidden_states[-1][-1][0,-1,:].detach().cpu()
    assert list(hidden.shape) == [3584] and torch.isfinite(hidden).all()
    token_limit = new_ids.numel() == max_new_tokens
    stop_reason = ('additional_tool_request' if CALL.search(new_text) else 'answer' if ANSWER.search(new_text)
                   else 'empty' if not new_text.strip() else 'format_failure')
    usable = stop_reason == 'answer' and 'addCriterion' not in new_text and not token_limit
    if verbose:
        print(f'[answer after fixed actual tool] {new_text!r}')
    audit = {'protocol': 'fixed_actual_scene_tool_answer_v1', 'stop_reason': stop_reason, 'usable': usable,
        'fixed_tool_call': fixed_call, 'fixed_tool_result': fixed_result, 'fixed_tool_origin': 'experiment-supplied action',
        'actual_tool_executions': 1, 'model_generated_initial_tool_actions': 0,
        'additional_tool_request': bool(CALL.search(new_text)),
        'rounds': [{'round1': 1, 'message_roles': [m['role'] for m in messages], 'rendered_text': rendered,
            'rendered_text_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
            'raw_text': new_text, 'token_limit': token_limit}],
        'resampled': False, 'state_rule': 'last generate, last decode step, last layer, last position'}
    return hidden, 1, 0, audit
