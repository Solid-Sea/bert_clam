"""
持续学习策略模式
定义统一的策略接口，实现模块化的持续学习组件
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class ContinualLearningStrategy(ABC):
    """持续学习策略抽象基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self,
              hidden_states: torch.Tensor,
              model_output: Dict[str, Any],
              task_id: int,
              **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        应用策略到隐藏状态
        
        Args:
            hidden_states: 当前隐藏状态 [batch, seq_len, hidden_size]
            model_output: 模型输出字典（包含 attentions, pooled_output 等）
            task_id: 当前任务ID
            **kwargs: 额外参数
            
        Returns:
            (enhanced_hidden_states, optional_loss)
        """
        pass


class EWCStrategy(ContinualLearningStrategy):
    """弹性权重巩固策略"""
    
    def __init__(self, ewc_module, model):
        super().__init__("EWC")
        self.ewc = ewc_module
        self.model = model
    
    def apply(self,
              hidden_states: torch.Tensor,
              model_output: Dict[str, Any],
              task_id: int,
              **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """EWC不修改hidden_states，只计算正则化损失"""
        ewc_loss = None
        if self.ewc and hasattr(self.ewc, 'compute_multi_task_ewc_loss'):
            task_memory = kwargs.get('task_memory', {})
            active_tasks = list(task_memory.keys())
            if active_tasks:
                ewc_loss = self.ewc.compute_multi_task_ewc_loss(self.model, active_tasks)
        
        return hidden_states, ewc_loss


class MRBStrategy(ContinualLearningStrategy):
    """记忆重放库策略"""
    
    def __init__(self, mrb_module, fusion_weight: float = 0.2):
        super().__init__("MRB")
        self.mrb = mrb_module
        self.fusion_weight = fusion_weight
    
    def apply(self,
              hidden_states: torch.Tensor,
              model_output: Dict[str, Any],
              task_id: int,
              **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """检索记忆并融合知识"""
        task_memory = kwargs.get('task_memory', {})
        
        if self.mrb and task_id in task_memory:
            retrieved_knowledge = self.mrb(hidden_states, task_id)
            fused_output = (1 - self.fusion_weight) * hidden_states + self.fusion_weight * retrieved_knowledge
            return fused_output, None
        
        return hidden_states, None


class ALPStrategy(ContinualLearningStrategy):
    """自适应LoRA池化策略"""
    
    def __init__(self, alp_module):
        super().__init__("ALP")
        self.alp = alp_module
    
    def apply(self,
              hidden_states: torch.Tensor,
              model_output: Dict[str, Any],
              task_id: int,
              **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """应用自适应LoRA池化"""
        task_embeddings = kwargs.get('task_embeddings', {})
        get_task_embedding = kwargs.get('get_task_embedding')
        input_ids = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask')
        
        if self.alp and task_id in task_embeddings and get_task_embedding is not None:
            task_embedding = get_task_embedding(input_ids, attention_mask)
            enhanced_output = self.alp(
                hidden_states,
                task_embedding,
                'classifier',
                task_id
            )
            return enhanced_output, None
        
        return hidden_states, None


class GrammarStrategy(ContinualLearningStrategy):
    """语法感知策略"""
    
    def __init__(self, grammar_module):
        super().__init__("Grammar")
        self.grammar = grammar_module
    
    def apply(self,
              hidden_states: torch.Tensor,
              model_output: Dict[str, Any],
              task_id: int,
              **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """应用语法感知增强"""
        if not self.grammar:
            return hidden_states, None
        
        # 获取注意力权重
        attentions = model_output.get('attentions')
        attention_weights = attentions[-1] if attentions else None
        
        # 应用语法感知
        if attention_weights is not None:
            enhanced_output = self.grammar(hidden_states, attention_weights)
        else:
            enhanced_output = self.grammar(hidden_states)
        
        # 计算语法损失
        grammar_loss = None
        if hasattr(self.grammar, 'compute_syntax_aware_loss'):
            grammar_loss = self.grammar.compute_syntax_aware_loss(enhanced_output)
        
        return enhanced_output, grammar_loss