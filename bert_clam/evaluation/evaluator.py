"""
BERT-CLAM评估器 - 扩展版
支持所有研究级指标
"""

import torch
import time
import psutil
from torch.utils.data import DataLoader
from typing import Dict, Any
from sklearn.metrics import f1_score, matthews_corrcoef
import numpy as np


class Evaluator:
    """简化的评估器，支持所有研究级指标"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def evaluate(self, model, data_loader: DataLoader, task_id: int = 0) -> Dict[str, Any]:
        """评估模型并返回所有指标"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0
        num_batches = 0
        
        # 记录内存和时间
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch.get('labels', batch.get('label')).to(model.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    task_id=task_id
                )
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    num_batches += 1
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # 计算指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_predictions == all_labels)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_predictions)
        
        return {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'matthews_correlation': float(mcc),
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'inference_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'num_samples': len(all_labels)
        }
    
    def compute_forgetting_metrics(self, performance_history: Dict[int, list]) -> Dict[str, float]:
        """计算遗忘指标"""
        forgetting_rates = {}
        
        for task_id, performances in performance_history.items():
            if len(performances) >= 2:
                max_acc = max(performances[:-1])
                final_acc = performances[-1]
                forgetting_rates[task_id] = max(0.0, max_acc - final_acc)
        
        avg_forgetting = np.mean(list(forgetting_rates.values())) if forgetting_rates else 0.0
        
        return {
            'forgetting_rates': forgetting_rates,
            'average_forgetting': float(avg_forgetting)
        }
    
    def compute_transfer_metrics(self, performance_matrix: Dict[int, Dict[int, float]]) -> Dict[str, float]:
        """计算迁移指标"""
        backward_transfers = []
        
        task_ids = sorted(performance_matrix.keys())
        for i in range(1, len(task_ids)):
            prev_task = task_ids[i-1]
            if prev_task in performance_matrix and len(performance_matrix[prev_task]) >= 2:
                performances = list(performance_matrix[prev_task].values())
                bwt = performances[-1] - performances[0]
                backward_transfers.append(bwt)
        
        return {
            'backward_transfer': float(np.mean(backward_transfers)) if backward_transfers else 0.0
        }


class ForgettingTracker:
    """遗忘追踪器（保持兼容性）"""
    
    def __init__(self):
        self.performance_history = {}
    
    def update(self, task_id: int, accuracy: float):
        if task_id not in self.performance_history:
            self.performance_history[task_id] = []
        self.performance_history[task_id].append(accuracy)
    
    def compute_forgetting(self, task_id: int) -> float:
        if task_id not in self.performance_history or len(self.performance_history[task_id]) < 2:
            return 0.0
        
        performances = self.performance_history[task_id]
        max_acc = max(performances[:-1])
        final_acc = performances[-1]
        
        return max(0.0, max_acc - final_acc)