"""测试记忆重放库模块"""
import unittest
import torch
import numpy as np
from bert_clam.core.memory_replay_bank import AdaptiveMemoryRetrieval, EnhancedAdaptiveMemoryRetrieval

class TestAdaptiveMemoryRetrieval(unittest.TestCase):
    """测试AdaptiveMemoryRetrieval类"""
    
    def setUp(self):
        """测试前准备"""
        self.dim = 768
        self.k = 5
        try:
            self.amr = AdaptiveMemoryRetrieval(dim=self.dim, k=self.k)
        except ImportError:
            self.skipTest("FAISS not available")
    
    def test_add_and_retrieve(self):
        """测试添加和检索功能"""
        task_id = 0
        batch_size = 4
        
        # 创建测试数据
        states = torch.randn(batch_size, self.dim)
        labels = torch.randint(0, 2, (batch_size,))
        
        # 添加到记忆库
        self.amr.add_memory(task_id, states, labels)
        
        # 验证记忆库大小
        self.assertEqual(self.amr.get_memory_size(), batch_size)
        
        # 检索知识
        query = torch.randn(2, self.dim)
        retrieved_states, retrieved_labels = self.amr.retrieve_knowledge(query, task_id)
        
        # 验证检索结果形状
        self.assertEqual(retrieved_states.shape[0], 2)
        self.assertEqual(retrieved_states.shape[1], self.dim)
        self.assertEqual(retrieved_labels.shape[0], 2)
    
    def test_distillation_loss(self):
        """测试知识蒸馏损失计算"""
        current = torch.randn(4, self.dim)
        teacher = torch.randn(4, self.dim)
        
        loss = self.amr.compute_distillation_loss(current, teacher)
        
        # 验证损失是标量且非负
        self.assertTrue(loss.dim() == 0)
        self.assertTrue(loss.item() >= 0)

class TestEnhancedAdaptiveMemoryRetrieval(unittest.TestCase):
    """测试EnhancedAdaptiveMemoryRetrieval类"""
    
    def setUp(self):
        """测试前准备"""
        try:
            self.eamr = EnhancedAdaptiveMemoryRetrieval(hidden_size=768, k=5)
        except ImportError:
            self.skipTest("FAISS not available")
    
    def test_forward(self):
        """测试前向传播"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output = self.eamr(hidden_states, task_id=0)
        
        # 验证输出形状
        self.assertEqual(output.shape, hidden_states.shape)
    
    def test_add_to_memory(self):
        """测试添加到记忆库"""
        hidden_states = torch.randn(4, 10, 768)
        labels = torch.randint(0, 2, (4,))
        
        self.eamr.add_to_memory(hidden_states, labels, task_id=0)
        
        # 验证记忆库已更新
        self.assertGreater(self.eamr.amr_core.get_memory_size(), 0)

if __name__ == '__main__':
    unittest.main()