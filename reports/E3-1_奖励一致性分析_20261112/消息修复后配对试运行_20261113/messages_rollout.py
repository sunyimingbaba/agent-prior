"""Independent rollout: explicit messages, closed tool tags, min-new-token guard."""
import hashlib
import re

import torch

CALL = re.compile(r'<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?')
ANSWER = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


def initial_messages(caption, system_prompt):
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': [{'type': 'text', 'text': '查询文本: There is a harbor with several ships.'}]},
        {'role': 'assistant', 'content': '<tool_call>scene_classify</tool_call>'},
        {'role': 'user', 'content': '<result>harbor (0.61), port (0.24), ship (0.08)</result>'},
        {'role': 'assistant', 'content': '<answer>图中是一座港口，与查询文本相符。</answer>'},
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '查询文本: '+caption}]}]


@torch.no_grad()
def rollout_messages(model, processor, image, caption, tools, max_rounds, max_new_tokens,
                     device, verbose=False, injected_h_end=False, system_prompt=None, temperature=0.0):
    assert system_prompt and not injected_h_end and max_rounds > 0
    messages = initial_messages(caption, system_prompt)
    used, filtered = 0, 0
    rounds, normalizations = [], []
    stop_reason = 'round_limit'
    for round_index in range(max_rounds):
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[rendered], images=[image], return_tensors='pt').to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1,
            do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.0,
            output_hidden_states=True, return_dict_in_generate=True)
        new_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        new_text = processor.decode(new_ids, skip_special_tokens=True)
        hidden = outputs.hidden_states[-1][-1][0,-1,:].detach().cpu()
        assert list(hidden.shape) == [3584] and torch.isfinite(hidden).all()
        rounds.append({'round1': round_index+1, 'message_roles': [m['role'] for m in messages],
            'rendered_text': rendered, 'rendered_text_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
            'raw_text': new_text, 'token_limit': new_ids.numel() == max_new_tokens})
        if verbose:
            print(f'[round {round_index+1}] {new_text!r}')
        match = CALL.search(new_text)
        if match is None or match.group(1) != 'scene_classify':
            stop_reason = 'answer' if ANSWER.search(new_text) else ('empty' if not new_text.strip() else 'format_failure')
            break
        result = tools['scene_classify'](image)
        if result is None:
            filtered += 1
            stop_reason = 'tool_filtered'
            break
        used += 1
        normalized = new_text[:match.start()]+'<tool_call>scene_classify</tool_call>'+new_text[match.end():]
        normalizations.append({'round1': round_index+1, 'raw_assistant_text': new_text,
            'feedback_assistant_text': normalized, 'tool_result': result})
        messages += [{'role': 'assistant', 'content': normalized},
                     {'role': 'user', 'content': '<result>'+result+'</result>'}]
    trajectory = '\n'.join(r['raw_text'] for r in rounds)
    usable = (stop_reason == 'answer' and bool(rounds[-1]['raw_text'].strip())
        and 'addCriterion' not in trajectory and not any(r['token_limit'] for r in rounds))
    return hidden, used, filtered, {'protocol': 'messages_closed_minnew1_v1', 'stop_reason': stop_reason,
        'usable': usable, 'rounds': rounds, 'tool_context_normalizations': normalizations,
        'resampled': False, 'state_rule': 'last generate, last decode step, last layer, last position'}
