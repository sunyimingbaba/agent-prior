import unittest
import torch
from supervised_format import answer_mask,masked_token_nll

class FormatSupervisionTests(unittest.TestCase):
    def test_body_and_crossing_tokens_masked(self):
        self.assertEqual(answer_mask([(0,2),(2,8),(8,12),(11,15),(15,19),(19,24)],8,15),[True,True,False,False,True,True])
    def test_only_selected_positions_have_direct_logit_gradients(self):
        logits=torch.tensor([[1.,2.,3.,4.],[4.,1.,0.,2.],[2.,4.,3.,1.]],requires_grad=True)
        target=torch.tensor([1,0,2]);mask=torch.tensor([True,False,True])
        loss,stats=masked_token_nll(logits,target,mask,[3]);loss.backward()
        self.assertTrue(torch.equal(logits.grad[1],torch.zeros(4)))
        self.assertEqual(float(logits.grad[0,3]),0.)
        self.assertEqual(stats['active_tokens'],2)
    def test_temperature_and_first_eos_match_manual_probabilities(self):
        logits=torch.tensor([[1.,2.,8.],[3.,1.,2.]])
        target=torch.tensor([1,2]);mask=torch.ones(2,dtype=torch.bool)
        loss,_=masked_token_nll(logits,target,mask,[2],temperature=.7)
        expected=-torch.log_softmax(logits[0,:2]/.7,dim=0)[1]-torch.log_softmax(logits[1]/.7,dim=0)[2]
        self.assertTrue(torch.allclose(loss,expected))
    def test_empty_body_span_all_format_tokens(self):
        self.assertEqual(answer_mask([(0,8),(8,17)],8,8),[True,True])

if __name__=='__main__':unittest.main()
