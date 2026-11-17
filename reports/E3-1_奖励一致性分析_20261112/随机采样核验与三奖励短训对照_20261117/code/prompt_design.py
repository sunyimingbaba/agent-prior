"""Three fixed candidate prompts; all choices are sealed before validation scoring."""
import hashlib
from stage_io import read
from milestone_common import ROOT

def messages(kind,case,system):
 query=read(ROOT/'plan.json')['base_query'];tail={'role':'user','content':[{'type':'image'},{'type':'text','text':'查询文本: '+query}]}
 if kind=='legacy':
  from live_rollout_legacy import messages_for
  return messages_for(query,system)
 if kind=='contract_only':
  text=system+'\n当前任务的交互顺序：第一回合只输出完整的 <tool_call>scene_classify</tool_call>。等待用户回合提供外部分类结果，再输出完整的 <answer>场景描述</answer>。外部工具结果的result标签由外部程序提供。'
  return [{'role':'system','content':text},tail]
 if kind=='neutral_placeholder':
  result='<result>[external scene classification output]</result>'
  answer='<answer>Describe the current image using its visible features and the external scene result.</answer>'
 elif kind=='rotated_actual3':
  examples=read(ROOT/'inputs/rotating_examples.json');i=int(hashlib.sha256(('protocol_rotate_20261116:'+case['image']).encode()).hexdigest()[:8],16)%3
  x=examples[i];result='<result>'+x['feedback']['text']+'</result>';answer='<answer>The scene classifier reports '+x['top1']+'.</answer>'
 else:raise ValueError(kind)
 return [{'role':'system','content':system},{'role':'user','content':[{'type':'text','text':'查询文本: '+query}]},{'role':'assistant','content':'<tool_call>scene_classify</tool_call>'},{'role':'user','content':result},{'role':'assistant','content':answer},tail]
