import unittest
import torch
from live_rollout_eos import ToolOnlyStop

class Processor:
    def __init__(self,text):self.text=text
    def decode(self,*args,**kwargs):return self.text

class EOSBoundaryTests(unittest.TestCase):
    def stop(self,text):return bool(ToolOnlyStop(Processor(text),0)(torch.tensor([[1]]),None)[0])
    def test_full_registered_tool_stops(self):self.assertTrue(self.stop('<tool_call>scene_classify</tool_call>'))
    def test_incomplete_and_unknown_calls_continue(self):
        self.assertFalse(self.stop('<tool_call>scene_classify'));self.assertFalse(self.stop('<tool_call>unknown</tool_call>'))
    def test_answer_closing_defers_to_native_eos(self):
        self.assertFalse(self.stop('<answer>A visible scene.</answer>'))
    def test_model_result_is_not_replaced_or_accepted(self):
        self.assertFalse(self.stop('<result>airport</result>'))

if __name__=='__main__':unittest.main()
