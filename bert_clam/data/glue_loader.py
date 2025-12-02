"""
GLUE数据加载器
"""

from .continual_dataset import ContinualDataset, TaskDataset
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer


class GLUECLDataLoader:
    """GLUE-CL数据加载器"""
    
    def __init__(self,
                 task_sequence: List[str],
                 tokenizer: AutoTokenizer,
                 batch_size: int = 32,
                 max_length: int = 512):
        self.task_sequence = task_sequence
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 创建持续学习数据集
        self.continual_dataset = ContinualDataset(
            task_names=task_sequence,
            tokenizer=tokenizer,
            max_length=max_length
        )
    
    def get_task_train_loader(self, task_name: str) -> DataLoader:
        """获取任务训练数据加载器"""
        return self.continual_dataset.get_task_dataloader(
            task_name, self.batch_size, 'train', shuffle=True
        )
    
    def get_task_val_loader(self, task_name: str) -> DataLoader:
        """获取任务验证数据加载器"""
        return self.continual_dataset.get_task_dataloader(
            task_name, self.batch_size, 'validation', shuffle=False
        )
    
    def get_task_test_loader(self, task_name: str) -> DataLoader:
        """获取任务测试数据加载器"""
        return self.continual_dataset.get_task_dataloader(
            task_name, self.batch_size, 'test', shuffle=False
        )
    
    def get_all_task_loaders(self) -> Dict[str, Dict[str, DataLoader]]:
        """获取所有任务的所有数据加载器"""
        all_loaders = {}
        
        for task_name in self.task_sequence:
            all_loaders[task_name] = {
                'train': self.get_task_train_loader(task_name),
                'validation': self.get_task_val_loader(task_name),
                'test': self.get_task_test_loader(task_name)
            }
        
        return all_loaders
    
    def get_task_names(self) -> List[str]:
        """获取任务名称列表"""
        return self.task_sequence