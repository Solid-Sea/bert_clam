"""
持续学习数据集模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datasets import Dataset as HFDataset


class TaskDataset(Dataset):
    """单任务数据集包装器"""
    
    def __init__(self, 
                 dataset: HFDataset,
                 tokenizer,
                 max_length: int = 512,
                 task_id: int = 0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_id = task_id
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # 初始化encoded变量
        encoded = None
        
        # Tokenize文本
        if 'sentence' in item:
            text = item['sentence']
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        elif 'text' in item:
            text = item['text']
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        elif 'premise' in item and 'hypothesis' in item:
            # 对于成对任务如MNLI
            text = item['premise']
            text_pair = item['hypothesis']
            encoded = self.tokenizer(
                text,
                text_pair=text_pair,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        elif 'question' in item and 'sentence' in item:
            # 对于问答任务
            text = item['question']
            text_pair = item['sentence']
            encoded = self.tokenizer(
                text,
                text_pair=text_pair,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        else:
            # 假设第一个文本字段是主要文本
            text_fields = ['question', 'sentence1', 'sentence2', 'text', 'premise', 'hypothesis']
            text = None
            for field in text_fields:
                if field in item:
                    text = item[field]
                    break
            
            if text is None:
                raise ValueError(f"No text field found in dataset item: {item.keys()}")
            
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        if encoded is None:
            raise ValueError(f"Failed to tokenize item: {item}")
        
        # 处理标签 - 支持多种标签字段名
        label = None
        label_fields = ['label', 'labels', 'idx', 'index', 'target', 'category']
        for field in label_fields:
            if field in item:
                label = item[field]
                break
        
        # 如果没找到标签字段，使用默认值
        if label is None:
            label = 0
        
        if isinstance(label, (list, np.ndarray)):
            label = label[0] if len(label) > 0 else 0
        elif isinstance(label, (str, float)):
            # 如果标签是字符串或浮点数，尝试转换为整数
            try:
                label = int(label)
            except (ValueError, TypeError):
                label = 0
        
        # 确保标签在有效范围内（对于2分类任务）
        label = max(0, min(1, int(label)))  # 假设是二分类任务
        
        # 返回字典格式的数据
        result = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'task_id': self.task_id,
            'label': torch.tensor(int(label), dtype=torch.long)
        }
        
        # 添加token_type_ids（如果存在）
        if 'token_type_ids' in encoded:
            result['token_type_ids'] = encoded['token_type_ids'].squeeze(0)
        
        return result


class ContinualDataset:
    """持续学习数据集管理器"""
    
    def __init__(self,
                 task_names: List[str],
                 tokenizer,
                 max_length: int = 512,
                 data_dir: str = None):
        self.task_names = task_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = data_dir
        
        # 存储每个任务的数据集
        self.task_datasets = {}
        self.task_sizes = {}
        
        # 当前任务索引
        self.current_task_idx = 0
    
    def load_task_dataset(self, task_name: str, split: str = 'train'):
        """加载特定任务的数据集"""
        from datasets import load_dataset
        
        # 加载GLUE任务数据集
        if task_name.lower() == 'cola':
            dataset = load_dataset('glue', 'cola', split=split)
        elif task_name.lower() == 'sst2':
            dataset = load_dataset('glue', 'sst2', split=split)
        elif task_name.lower() == 'mrpc':
            dataset = load_dataset('glue', 'mrpc', split=split)
        elif task_name.lower() == 'qqp':
            dataset = load_dataset('glue', 'qqp', split=split)
        elif task_name.lower() == 'mnli':
            dataset = load_dataset('glue', 'mnli', split=split if split != 'validation' else 'validation_matched')
        elif task_name.lower() == 'qnli':
            dataset = load_dataset('glue', 'qnli', split=split)
        elif task_name.lower() == 'rte':
            dataset = load_dataset('glue', 'rte', split=split)
        elif task_name.lower() == 'wnli':
            dataset = load_dataset('glue', 'wnli', split=split)
        elif task_name.lower() == 'stsb':
            dataset = load_dataset('glue', 'stsb', split=split)
        else:
            # 默认加载GLUE任务
            try:
                dataset = load_dataset('glue', task_name.lower(), split=split)
            except:
                raise ValueError(f"Unsupported task: {task_name}")
        
        # 创建任务特定的数据集
        task_id = len(self.task_datasets)
        task_dataset = TaskDataset(
            dataset=dataset,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            task_id=task_id
        )
        
        self.task_datasets[task_name] = task_dataset
        self.task_sizes[task_name] = len(task_dataset)
        
        return task_dataset
    
    def get_task_by_id(self, task_id: int, split: str = 'train'):
        """通过ID获取任务"""
        if task_id < len(self.task_names):
            task_name = self.task_names[task_id]
            train_loader = self.get_task_dataloader(task_name, 32, 'train', shuffle=True)
            val_loader = self.get_task_dataloader(task_name, 32, 'validation', shuffle=False)
            test_loader = self.get_task_dataloader(task_name, 32, 'test', shuffle=False)
            
            return task_name, train_loader, val_loader, test_loader
        return None
    
    def get_task_info(self, task_id: int) -> Dict[str, Any]:
        """获取任务信息"""
        if task_id < len(self.task_names):
            return {'name': self.task_names[task_id]}
        return None
    
    def get_task_dataloader(self, 
                           task_name: str, 
                           batch_size: int = 32,
                           split: str = 'train',
                           shuffle: bool = True) -> DataLoader:
        """获取特定任务的数据加载器"""
        if task_name not in self.task_datasets:
            self.load_task_dataset(task_name, split)
        
        dataset = self.task_datasets[task_name]
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
    
    def get_task_sequence(self, 
                         task_names: List[str] = None,
                         batch_size: int = 32,
                         splits: List[str] = ['train', 'validation']) -> Dict[str, DataLoader]:
        """获取任务序列的数据加载器"""
        if task_names is None:
            task_names = self.task_names
        
        task_loaders = {}
        
        for task_name in task_names:
            for split in splits:
                loader_name = f"{task_name}_{split}"
                task_loaders[loader_name] = self.get_task_dataloader(
                    task_name, batch_size, split, 
                    shuffle=(split == 'train')
                )
        
        return task_loaders
    
    def get_next_task(self, batch_size: int = 32) -> Optional[Tuple[str, DataLoader]]:
        """获取下一个任务"""
        if self.current_task_idx < len(self.task_names):
            task_name = self.task_names[self.current_task_idx]
            self.current_task_idx += 1
            
            train_loader = self.get_task_dataloader(task_name, batch_size, 'train', shuffle=True)
            val_loader = self.get_task_dataloader(task_name, batch_size, 'validation', shuffle=False)
            
            return task_name, train_loader, val_loader
        
        return None
    
    def reset_task_counter(self):
        """重置任务计数器"""
        self.current_task_idx = 0
    
    def get_num_tasks(self) -> int:
        """获取任务数量"""
        return len(self.task_names)
    
    def get_task_size(self, task_name: str) -> int:
        """获取特定任务的大小"""
        if task_name in self.task_sizes:
            return self.task_sizes[task_name]
        else:
            self.load_task_dataset(task_name, 'train')
            return self.task_sizes[task_name]


class GLUECLDataLoader:
    """GLUE-CL数据加载器"""
    
    def __init__(self,
                 task_sequence: List[str],
                 tokenizer,
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