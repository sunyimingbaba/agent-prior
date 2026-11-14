"""Software tests of token masks and actual tool-feedback message routing."""
import copy
from types import SimpleNamespace
import unittest
import torch

from live_rollout import effective_response, messages_for, rollout, assistant_logprob, ToolBoundaryStop


class Inputs(dict):
    @property
    def input_ids(self):return self['input_ids']
    def to(self, device):return self


class Processor:
    def __init__(self):self.messages=[]
    def apply_chat_template(self,messages,**kwargs):
        self.messages.append(copy.deepcopy(messages))
        return 'test-context<|im_start|>assistant\n'
    def __call__(self,**kwargs):return Inputs(input_ids=torch.tensor([[5,6,7]]))
    def decode(self,ids,**kwargs):
        return {10:'<tool_call>scene_classify</tool_call>',11:'<answer>visible scene</answer>',
                12:'<result>fabricated</result><answer>x</answer>'}[ids[0]]


class Model:
    def __init__(self, ids):
        self.training=False;self.ids=iter(ids);self.generation_config=SimpleNamespace(eos_token_id=[2])
    def eval(self):self.training=False;return self
    def train(self,mode=True):self.training=mode;return self
    def generate(self,**kwargs):
        ids=next(self.ids)
        return SimpleNamespace(sequences=torch.tensor([[5,6,7]+ids]),
            hidden_states=[(torch.full((1,1,3584),float(i)),) for i in range(len(ids))])


class ContractTests(unittest.TestCase):
    def test_initial_eos_mask_matches_sampling(self):
        logits=torch.tensor([[9.,1.,2.],[1.,2.,3.]],requires_grad=True)
        target=torch.tensor([2,0])
        actual=assistant_logprob(logits,target,[0])
        expected=torch.log_softmax(torch.tensor([1.,2.]),0)[1]+torch.log_softmax(torch.tensor([1.,2.,3.]),0)[0]
        self.assertTrue(torch.allclose(actual,expected))
        actual.backward()
        self.assertEqual(float(logits.grad[0,0]),0.)
    def test_registered_tool_boundary_stops(self):
        stop=ToolBoundaryStop(Processor(),3)
        self.assertTrue(bool(stop(torch.tensor([[5,6,7,10]]),None)[0]))
        self.assertFalse(bool(stop(torch.tensor([[5,6,7,11]]),None)[0]))
    def test_eos_inclusive_and_padding_excluded(self):
        self.assertEqual(effective_response([4,5,2,0,0],[2],5),[4,5,2])
        self.assertEqual(effective_response([4,5,0,0],[2,0],2),[4,5])
        self.assertEqual(effective_response([2,0],[2],2),[2])
        self.assertEqual(effective_response([4,5,6],[2],3),[4,5,6])
    def test_explicit_plain_system(self):
        self.assertEqual(messages_for('x','plain',False)[0]['content'],'plain')
        self.assertEqual(len(messages_for('x','plain',False)),2)
    def test_actual_tool_feedback_and_hidden_state(self):
        processor=Processor();calls=[]
        def scene(image):calls.append(image);return {'text':'real-scene-1','confidence':.9,'accepted':True}
        hidden,record=rollout(Model([[10,2],[11,2]]),processor,'image','caption',{'scene_classify':scene},
                              'system',{'max_rounds':3,'max_new_tokens_per_round':256})
        self.assertEqual(calls,['image'])
        self.assertTrue(record['quality_usable'])
        self.assertEqual(record['tools_used'],1)
        self.assertEqual(processor.messages[1][-2]['role'],'assistant')
        self.assertEqual(processor.messages[1][-1],{'role':'user','content':'<result>real-scene-1</result>'})
        self.assertTrue(torch.equal(hidden,torch.ones(3584)))
        self.assertEqual(record['rounds'][-1]['state_input_token_id'],11)
    def test_generated_tool_result_is_flagged(self):
        _,record=rollout(Model([[12,2]]),Processor(),'image','caption',{},'system',
                          {'max_rounds':3,'max_new_tokens_per_round':256})
        self.assertFalse(record['quality_usable'])
        self.assertEqual(record['stop_reason'],'model_generated_tool_result')


if __name__=='__main__':unittest.main()
