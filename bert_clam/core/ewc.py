"""
EWC: 弹性权重巩固模块
防止灾难性遗忘的正则化机制
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import copy


class ElasticWeightConsolidation:
    """弹性权重巩固 - 防止灾难性遗忘"""
    
    def __init__(self, lambda_ewc: float = 0.15, fisher_samples: int = 1000):
        """
        Args:
            lambda_ewc: EWC正则化系数
            fisher_samples: Fisher信息矩阵采样大小
        """
        self.lambda_ewc = lambda_ewc
        self.fisher_samples = fisher_samples
        self.fisher_dict = {}
        self.old_params_dict = {}
        self.task_count = 0
    
    def compute_fisher(self, model: nn.Module, dataloader: DataLoader,
                      task_id: int, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """计算Fisher信息矩阵
        
        Args:
            model: 模型
            dataloader: 数据加载器
            num_samples: 采样数量，默认使用配置值
            
        Returns:
            fisher_dict: Fisher信息矩阵字典
        """
        if num_samples is None:
            num_samples = self.fisher_samples
            
        fisher_dict = {}
        
        # 初始化Fisher矩阵
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
        
        # 保持训练模式以支持反向传播
        model.train()
        samples_seen = 0
        device = next(model.parameters()).device
        
        for batch in dataloader:
            if samples_seen >= num_samples:
                break

            # 标准化批处理数据处理
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
                labels = batch['labels'].to(device)
            except (KeyError, AttributeError):
                print("EWC compute_fisher: Batch format not as expected. Skipping batch.")
                continue

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                task_id=task_id  # 传递task_id
            )
            
            # 获取logits
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('logits'))
            else:
                logits = outputs
            
            # 计算损失
            if labels is not None:
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                if labels.dim() > 1:
                    labels = labels.view(-1)
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = -logits.mean()
            
            # 反向传播
            model.zero_grad()
            loss.backward()
            
            # 累积梯度平方
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            if isinstance(input_ids, torch.Tensor):
                samples_seen += input_ids.size(0)
            else:
                # Fallback for non-tensor inputs, though less likely with new logic
                samples_seen += 1
        
        # 归一化
        for name in fisher_dict:
            fisher_dict[name] /= max(samples_seen, 1)
        
        return fisher_dict
    
    def save_task_checkpoint(self, model: nn.Module, dataloader: DataLoader, 
                           task_id: int = None, num_samples: int = None):
        """保存当前任务的参数和Fisher信息
        
        Args:
            model: 当前模型
            dataloader: 任务数据加载器
            task_id: 任务ID（用于跟踪）
            num_samples: Fisher计算采样数
        """
        # 计算Fisher信息
        self.fisher_dict = self.compute_fisher(model, dataloader, num_samples)
        
        # 保存当前参数
        self.old_params_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.old_params_dict[name] = param.data.clone()
        
        if task_id is not None:
            print(f"Saved EWC checkpoint for task {task_id}")
        
        self.task_count += 1
    
    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """计算EWC正则化损失
        
        Args:
            model: 当前模型
            
        Returns:
            ewc_loss: EWC损失
        """
        if not self.fisher_dict or not self.old_params_dict:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if (name in self.fisher_dict and 
                name in self.old_params_dict and 
                param.requires_grad):
                fisher = self.fisher_dict[name]
                old_param = self.old_params_dict[name]
                
                # EWC损失：Fisher * (θ - θ_old)^2
                param_diff = param - old_param
                ewc_loss += (fisher * param_diff ** 2).sum()
        
        return self.lambda_ewc * ewc_loss
    
    def update_fisher_matrix(self, model: nn.Module, new_fisher: Dict[str, torch.Tensor],
                           update_factor: float = 0.5):
        """更新Fisher矩阵（用于任务序列学习）
        
        Args:
            model: 当前模型
            new_fisher: 新的Fisher矩阵
            update_factor: 更新因子
        """
        for name in self.fisher_dict:
            if name in new_fisher:
                self.fisher_dict[name] = (1 - update_factor) * self.fisher_dict[name] + \
                                       update_factor * new_fisher[name]
    
    def clear_checkpoint(self):
        """清空检查点"""
        self.fisher_dict.clear()
        self.old_params_dict.clear()
        self.task_count = 0


class EnhancedElasticWeightConsolidation(nn.Module):
    """增强版弹性权重巩固模块"""
    
    def __init__(self, lambda_ewc: float = 0.15, fisher_samples: int = 100,
                 update_strategy: str = "cumulative"):
        super().__init__()
        self.ewc_core = ElasticWeightConsolidation(lambda_ewc, fisher_samples)
        self.update_strategy = update_strategy
        self.task_fishers = {}  # 存储每个任务的Fisher矩阵
        self.task_params = {}   # 存储每个任务的参数
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """前向传播 - 返回EWC损失"""
        return self.ewc_core.compute_ewc_loss(model)
    
    def save_task_data(self, model: nn.Module, dataloader: DataLoader,
                      task_id: int, num_samples: int = None):
        """保存特定任务的数据"""
        fisher = self.ewc_core.compute_fisher(model, dataloader, task_id, num_samples)
        params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone()
        
        self.task_fishers[task_id] = fisher
        self.task_params[task_id] = params
    
    def compute_multi_task_ewc_loss(self, model: nn.Module, 
                                  active_tasks: list) -> torch.Tensor:
        """计算多任务EWC损失"""
        if not active_tasks:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for task_id in active_tasks:
            if (task_id in self.task_fishers and 
                task_id in self.task_params):
                
                fisher = self.task_fishers[task_id]
                old_params = self.task_params[task_id]
                
                task_loss = torch.tensor(0.0, device=next(model.parameters()).device)
                
                for name, param in model.named_parameters():
                    if (name in fisher and name in old_params and 
                        param.requires_grad):
                        param_diff = param - old_params[name]
                        task_loss += (fisher[name] * param_diff ** 2).sum()
                
                total_loss += task_loss
        
        return self.ewc_core.lambda_ewc * total_loss
    
    def get_task_importance(self, task_id: int) -> Dict[str, torch.Tensor]:
        """获取任务重要性（Fisher矩阵）"""
        return self.task_fishers.get(task_id, {})