"""测试策略模块"""
import unittest
import torch
import torch.nn as nn
from bert_clam.core.strategy import EWCStrategy, MRBStrategy, GrammarStrategy
from bert_clam.core.ewc import EnhancedElasticWeightConsolidation
from bert_clam.core.grammar_aware import EnhancedGrammarAwareModule

class DummyModel(nn.Module):
    """虚拟模型用于测试"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 2)

class DummyMRB(nn.Module):
    """虚拟MRB模块"""
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states, task_id):
        return hidden_states

class TestEWCStrategy(unittest.TestCase):
    """测试EWC策略"""
    
    def setUp(self):
        """测试前准备"""
        self.model = DummyModel()
        self.ewc = EnhancedElasticWeightConsolidation(lambda_ewc=0.5)
        self.strategy = EWCStrategy(self.ewc, self.model)
    
    def test_apply(self):
        """测试策略应用"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        model_output = {'logits': torch.randn(batch_size, 2)}
        
        enhanced_states, loss = self.strategy.apply(
            hidden_states, model_output, task_id=0
        )
        
        # 验证输出形状不变
        self.assertEqual(enhanced_states.shape, hidden_states.shape)
        # EWC策略不修改hidden_states
        self.assertTrue(torch.equal(enhanced_states, hidden_states))

class TestMRBStrategy(unittest.TestCase):
    """测试MRB策略"""
    
    def setUp(self):
        """测试前准备"""
        self.mrb = DummyMRB()
        self.strategy = MRBStrategy(self.mrb, fusion_weight=0.2)
    
    def test_apply(self):
        """测试策略应用"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        model_output = {}
        
        enhanced_states, loss = self.strategy.apply(
            hidden_states, model_output, task_id=0,
            task_memory={0: True}
        )
        
        # 验证输出形状
        self.assertEqual(enhanced_states.shape, hidden_states.shape)
        # 验证无损失返回
        self.assertIsNone(loss)

class TestGrammarStrategy(unittest.TestCase):
    """测试语法感知策略"""
    
    def setUp(self):
        """测试前准备"""
        self.grammar = EnhancedGrammarAwareModule(hidden_size=768)
        self.strategy = GrammarStrategy(self.grammar)
    
    def test_apply(self):
        """测试策略应用"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # 创建注意力权重
        num_heads = 12
        attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        model_output = {'attentions': (attention_weights,)}
        
        enhanced_states, loss = self.strategy.apply(
            hidden_states, model_output, task_id=0
        )
        
        # 验证输出形状
        self.assertEqual(enhanced_states.shape, hidden_states.shape)

if __name__ == '__main__':
    unittest.main()