"""测试EWC模块"""
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from bert_clam.core.ewc import ElasticWeightConsolidation, EnhancedElasticWeightConsolidation

class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, input_ids, attention_mask=None, labels=None, task_id=0):
        logits = self.fc(input_ids.float())
        return {'logits': logits}

class TestElasticWeightConsolidation(unittest.TestCase):
    """测试ElasticWeightConsolidation类"""
    
    def setUp(self):
        """测试前准备"""
        self.model = SimpleModel()
        self.ewc = ElasticWeightConsolidation(lambda_ewc=0.5, fisher_samples=10)
        
        # 创建测试数据
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(input_ids, labels)
        self.dataloader = DataLoader(
            [{'input_ids': x, 'labels': y} for x, y in dataset],
            batch_size=4
        )
    
    def test_compute_fisher(self):
        """测试Fisher信息矩阵计算"""
        fisher = self.ewc.compute_fisher(self.model, self.dataloader, task_id=0, num_samples=10)
        
        # 验证Fisher矩阵
        self.assertIsInstance(fisher, dict)
        self.assertGreater(len(fisher), 0)
        
        # 验证每个参数都有对应的Fisher值
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIn(name, fisher)
                self.assertEqual(fisher[name].shape, param.shape)
    
    def test_ewc_penalty(self):
        """测试EWC惩罚项计算"""
        # 保存检查点
        self.ewc.save_task_checkpoint(self.model, self.dataloader, task_id=0, num_samples=10)
        
        # 修改模型参数
        for param in self.model.parameters():
            param.data += 0.1
        
        # 计算EWC损失
        ewc_loss = self.ewc.compute_ewc_loss(self.model)
        
        # 验证损失非零
        self.assertTrue(ewc_loss.item() > 0)

class TestEnhancedElasticWeightConsolidation(unittest.TestCase):
    """测试EnhancedElasticWeightConsolidation类"""
    
    def setUp(self):
        """测试前准备"""
        self.model = SimpleModel()
        self.ewc = EnhancedElasticWeightConsolidation(lambda_ewc=0.5, fisher_samples=10)
        
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(input_ids, labels)
        self.dataloader = DataLoader(
            [{'input_ids': x, 'labels': y} for x, y in dataset],
            batch_size=4
        )
    
    def test_multi_task_ewc(self):
        """测试多任务EWC损失"""
        # 保存两个任务的数据
        self.ewc.save_task_data(self.model, self.dataloader, task_id=0, num_samples=10)
        self.ewc.save_task_data(self.model, self.dataloader, task_id=1, num_samples=10)
        
        # 计算多任务损失
        loss = self.ewc.compute_multi_task_ewc_loss(self.model, [0, 1])
        
        # 验证损失
        self.assertTrue(loss.item() >= 0)

if __name__ == '__main__':
    unittest.main()