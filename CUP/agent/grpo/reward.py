"""
阶段 C：GRPO 奖励函数。

总奖励 = R_format + λ · R_retrieval（λ 默认 1.0，由 --lam 透传）

R_format（格式奖励，常数）：
  - 轨迹含合法 <tool_call>xxx</tool_call> 且 <answer> 完整：+0.2
  - 无工具调用但 <answer> 完整：+0.2（纯推理轨迹合法）
  - 格式非法（乱文/未闭合标签）：0

R_retrieval（检索奖励，由 --reward-mode 控制）：
  - soft：1/(1+rank) 连续软分数（rank 0 → 1.0，rank 9 → 0.1）
  - hard：rank 命中（rank=0）给 1.0，未命中给 0
  - format-only：恒为 0（检索奖励完全不参与，等价 λ=0）
"""
import re

TOOL_CALL_PAT = re.compile(r"<tool_call>\s*([\w_]+)\s*(?:</tool_call>)?")
ANSWER_PAT = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def r_format(text: str) -> float:
    """格式奖励：合法 tool_call+完整 answer 或 纯 answer 都 +0.2，否则 0。"""
    has_tool = TOOL_CALL_PAT.search(text) is not None
    has_answer = ANSWER_PAT.search(text) is not None
    if has_answer and (has_tool or not has_tool):
        return 0.2
    return 0.0


def r_retrieval(rank: int, mode: str = "soft") -> float:
    """检索奖励：soft = 1/(1+rank)；hard = 命中 1.0 / 未命中 0；format-only = 恒 0。"""
    if mode == "soft":
        return 1.0 / (1.0 + rank)
    if mode == "hard":
        return 1.0 if rank == 0 else 0.0
    if mode == "format-only":
        return 0.0
    raise ValueError(f"未知 reward-mode: {mode}（可选 soft/hard/format-only）")


def total_reward(text: str, rank: int, lam: float = 1.0, mode: str = "soft") -> float:
    """总奖励 = R_format + λ · R_retrieval。"""
    return r_format(text) + lam * r_retrieval(rank, mode)
