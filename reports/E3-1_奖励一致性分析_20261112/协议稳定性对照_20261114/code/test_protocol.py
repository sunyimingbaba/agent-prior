import unittest
import torch
from live_rollout_protocol import FullProtocolStop,STRICT_TOOL,STRICT_ANSWER,messages_for

class Processor:
    def __init__(self,text):self.text=text
    def decode(self,*a,**k):return self.text

class ProtocolTests(unittest.TestCase):
    def stopped(self,text):return bool(FullProtocolStop(Processor(text),0)(torch.tensor([[1]]),None)[0])
    def test_incomplete_tool_continues(self):
        for text in ['<tool_call>scene_classify','<tool_call>scene_classify</tool','<tool_call>unknown</tool_call>']:
            self.assertFalse(self.stopped(text))
    def test_complete_registered_tool_stops(self):
        self.assertTrue(self.stopped('<tool_call>scene_classify</tool_call>'))
    def test_answer_must_be_complete_nonempty(self):
        self.assertTrue(self.stopped('<answer>A scene.</answer>'))
        for text in ['<answer>A scene.','<answer> </answer>','A scene.']:
            self.assertFalse(self.stopped(text))
    def test_model_result_is_preserved_as_failure(self):
        self.assertFalse(self.stopped('<result>airport (0.9)</result>'))
        self.assertFalse(self.stopped('<tool_call>scene_classify</tool_call><result>airport</result>'))
    def test_extra_text_stays_visible(self):
        self.assertFalse(self.stopped('Extra <answer>A scene.</answer>'))
    def test_neutral_last_query_and_role_order(self):
        x={'system':'s','example_user':'neutral','example_call':'c','example_result':'r','example_answer':'a'}
        m=messages_for('Describe the visible scene.','ignored','neutral_contract',x)
        self.assertEqual([z['role'] for z in m],['system','user','assistant','user','assistant','user'])
        self.assertEqual(m[-1]['content'][-1]['text'],'查询文本: Describe the visible scene.')

if __name__=='__main__':unittest.main()
