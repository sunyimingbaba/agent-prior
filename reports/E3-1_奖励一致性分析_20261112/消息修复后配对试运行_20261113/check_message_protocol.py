"""Bounded fake-model checks for feedback roles, tag repair and quality flags."""
import json
from types import SimpleNamespace
import torch
from messages_rollout import rollout_messages


class Batch(dict):
    def __getattr__(self, key):
        return self[key]

    def to(self, device):
        return self


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages.copy()
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, **kwargs):
        return Batch(input_ids=torch.tensor([[9]]), attention_mask=torch.ones(1,1,dtype=torch.long))

    def decode(self, ids, **kwargs):
        return self.texts[int(ids[0])]


class Model:
    def __init__(self, ids):
        self.ids = iter(ids)

    def generate(self, **kwargs):
        assert kwargs['min_new_tokens'] == 1
        return SimpleNamespace(sequences=torch.tensor([[9,next(self.ids)]]),
            hidden_states=((torch.ones(1,1,3584,dtype=torch.bfloat16),),))


p = Processor()
p.texts = {1:'<tool_call>scene_classify',2:'<answer>ok</answer>',3:'scene_classify',4:''}
tools = {'scene_classify': lambda _: 'airplane (0.90)'}
h, used, filtered, audit = rollout_messages(Model([1,2]),p,None,'caption',tools,
    max_rounds=3,max_new_tokens=64,device='cpu',system_prompt='test',temperature=1.0)
assert audit['usable'] and audit['stop_reason']=='answer' and used==1 and filtered==0
assert p.messages[-2] == {'role':'assistant','content':'<tool_call>scene_classify</tool_call>'}
assert p.messages[-1] == {'role':'user','content':'<result>airplane (0.90)</result>'}
for ids, reason in [([3],'format_failure'),([4],'empty'),([1],'round_limit')]:
    _,_,_, audit = rollout_messages(Model(ids),p,None,'caption',tools,max_rounds=1,
        max_new_tokens=64,device='cpu',system_prompt='test',temperature=1.0)
    assert not audit['usable'] and audit['stop_reason']==reason
print('MESSAGE_PROTOCOL_CHECKS_PASSED')
