"""Check actual tool execution, supplied roles, one continuation and failure flags."""
import json
from types import SimpleNamespace
import torch
from fixed_tool_rollout import rollout_fixed_tool


class Batch(dict):
    def __getattr__(self, key):
        return self[key]

    def to(self, device):
        return self


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, **kwargs):
        return Batch(input_ids=torch.tensor([[9]]), attention_mask=torch.ones(1,1,dtype=torch.long))

    def decode(self, ids, **kwargs):
        return self.text


class Model:
    calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        assert kwargs['min_new_tokens'] == 1
        return SimpleNamespace(sequences=torch.tensor([[9,1]]),
            hidden_states=((torch.ones(1,1,3584,dtype=torch.bfloat16),),))


for text, reason, usable in [('<answer>ok</answer>','answer',True),
    ('','empty',False),('scene_classify','format_failure',False),
    ('<tool_call>scene_classify','additional_tool_request',False),
    ('<answer>addCriterion</answer>','answer',False)]:
    p, model, events = Processor(), Model(), []
    p.text = text
    def tool(image):
        events.append('executed')
        return 'airplane (0.90)'
    h, used, filtered, audit = rollout_fixed_tool(model,p,None,'caption',{'scene_classify':tool},
        max_rounds=1,max_new_tokens=64,device='cpu',system_prompt='test',temperature=1.0)
    assert len(events)==used==model.calls==1 and filtered==0
    assert audit['stop_reason']==reason and audit['usable']==usable
    assert p.messages[-2] == {'role':'assistant','content':'<tool_call>scene_classify</tool_call>'}
    assert p.messages[-1] == {'role':'user','content':'<result>airplane (0.90)</result>'}
print('FIXED_TOOL_PROTOCOL_CHECKS_PASSED')
