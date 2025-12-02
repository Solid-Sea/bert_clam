"""
遗忘率评估器 - 专门用于评估持续学习中的遗忘现象
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class ForgettingEvaluator:
    """遗忘率评估器 - 用于评估持续学习中的遗忘现象"""
    
    def __init__(self):
        # 存储每个任务在每个训练阶段的性能
        self.task_performance_history = defaultdict(dict)  # {task_id: {stage: accuracy}}
        self.task_names = []  # 任务名称列表
        self.current_stage = 0  # 当前训练阶段
        
    def record_task_performance(self, task_id: int, stage_name: str, accuracy: float):
        """记录特定任务在特定阶段的性能"""
        self.task_performance_history[task_id][stage_name] = accuracy
    
    def record_task_performance_by_name(self, task_name: str, stage_name: str, accuracy: float):
        """按任务名称记录性能"""
        # 将任务名称映射到ID（简单映射，实际使用时可更复杂）
        task_id = self.task_names.index(task_name) if task_name in self.task_names else len(self.task_names)
        if task_name not in self.task_names:
            self.task_names.append(task_name)
        
        self.task_performance_history[task_id][stage_name] = accuracy
    
    def compute_forgetting_rate(self, task_id: int) -> Optional[float]:
        """计算特定任务的遗忘率"""
        if task_id not in self.task_performance_history:
            return None
        
        stages = self.task_performance_history[task_id]
        if len(stages) < 2:
            return 0.0  # 需要至少两个阶段才能计算遗忘
        
        # 获取所有阶段的准确率，按训练顺序排序
        stage_names = sorted(stages.keys())
        accuracies = [stages[stage] for stage in stage_names]
        
        if len(accuracies) < 2:
            return 0.0
        
        # 计算最高准确率（除了最终阶段）和最终准确率
        max_acc_before_final = max(accuracies[:-1])  # 不包括最终阶段的最高准确率
        final_acc = accuracies[-1]  # 最终准确率
        
        # 计算遗忘率：最高准确率 - 最终准确率
        forgetting = max(0.0, max_acc_before_final - final_acc)
        return forgetting
    
    def compute_forgetting_rate_by_name(self, task_name: str) -> Optional[float]:
        """按任务名称计算遗忘率"""
        if task_name not in self.task_names:
            return None
        
        task_id = self.task_names.index(task_name)
        return self.compute_forgetting_rate(task_id)
    
    def compute_average_forgetting(self) -> float:
        """计算所有任务的平均遗忘率"""
        forgetting_rates = []
        
        for task_id in range(len(self.task_names)):
            forgetting = self.compute_forgetting_rate(task_id)
            if forgetting is not None:
                forgetting_rates.append(forgetting)
        
        return np.mean(forgetting_rates) if forgetting_rates else 0.0
    
    def compute_backward_transfer(self, task_order: List[str]) -> float:
        """计算向后迁移 (Backward Transfer)
        
        BWT(t) = acc(t|model_t) - acc(t|model_0)
        """
        if len(task_order) < 2:
            return 0.0
        
        bwt_values = []
        
        for i in range(1, len(task_order)):
            current_task = task_order[i]
            prev_task = task_order[i-1]
            
            # 获取前一个任务在不同阶段的性能
            prev_task_id = self.task_names.index(prev_task) if prev_task in self.task_names else -1
            if prev_task_id == -1:
                continue
                
            stages = self.task_performance_history[prev_task_id]
            stage_names = sorted(stages.keys())
            
            if len(stage_names) >= 2:
                # 前一个任务在刚学会时的准确率（第i个阶段）
                initial_acc = stages[stage_names[i-1]] if i-1 < len(stage_names) else stages[stage_names[0]]
                
                # 前一个任务在学习当前任务后的准确率（最后一个阶段）
                final_acc = stages[stage_names[-1]]
                
                bwt = final_acc - initial_acc
                bwt_values.append(bwt)
        
        return np.mean(bwt_values) if bwt_values else 0.0
    
    def compute_forward_transfer(self, task_order: List[str]) -> float:
        """计算前向迁移 (Forward Transfer)
        
        FWT(t) = acc(t|model_{t-1}) - acc(t|model_0)
        """
        if len(task_order) < 2:
            return 0.0
        
        fwt_values = []
        
        for i in range(1, len(task_order)):
            current_task = task_order[i]
            current_task_id = self.task_names.index(current_task) if current_task in self.task_names else -1
            if current_task_id == -1:
                continue
            
            stages = self.task_performance_history[current_task_id]
            stage_names = sorted(stages.keys())
            
            if len(stage_names) >= 2:
                # 当前任务在只学习自己时的准确率（第一个阶段）
                initial_acc = stages[stage_names[0]]
                
                # 当前任务在学习完前面所有任务后的准确率（第i个阶段）
                learned_acc = stages[stage_names[min(i, len(stage_names)-1)]] if i < len(stage_names) else stages[stage_names[-1]]
                
                fwt = learned_acc - initial_acc
                fwt_values.append(fwt)
        
        return np.mean(fwt_values) if fwt_values else 0.0
    
    def get_comprehensive_metrics(self, task_order: List[str]) -> Dict[str, float]:
        """获取综合评估指标"""
        metrics = {}
        
        # 各任务遗忘率
        task_forgetting = {}
        for task_name in task_order:
            if task_name in self.task_names:
                forgetting = self.compute_forgetting_rate_by_name(task_name)
                if forgetting is not None:
                    task_forgetting[task_name] = forgetting
        
        metrics['task_forgetting_rates'] = task_forgetting
        metrics['average_forgetting_rate'] = self.compute_average_forgetting()
        metrics['backward_transfer'] = self.compute_backward_transfer(task_order)
        metrics['forward_transfer'] = self.compute_forward_transfer(task_order)
        
        return metrics
    
    def export_results(self, filepath: str):
        """导出评估结果"""
        results = {
            'task_performance_history': dict(self.task_performance_history),
            'task_names': self.task_names,
            'current_stage': self.current_stage,
            'comprehensive_metrics': self.get_comprehensive_metrics(self.task_names)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    def print_detailed_analysis(self):
        """打印详细分析"""
        print("遗忘率详细分析:")
        print("=" * 60)
        
        for i, task_name in enumerate(self.task_names):
            if i in self.task_performance_history:
                stages = self.task_performance_history[i]
                stage_names = sorted(stages.keys())
                accuracies = [stages[stage] for stage in stage_names]
                
                forgetting = self.compute_forgetting_rate(i)
                max_acc = max(accuracies) if accuracies else 0.0
                final_acc = accuracies[-1] if accuracies else 0.0
                
                print(f"任务 {task_name}:")
                print(f"  所有阶段准确率: {accuracies}")
                print(f"  最高准确率: {max_acc:.4f}")
                print(f"  最终准确率: {final_acc:.4f}")
                print(f"  遗忘率: {forgetting:.4f}")
                print(f"  准确率下降: {max_acc - final_acc:.4f}")
                print()


class ForgettingTracker:
    """遗忘跟踪器 - 用于在训练过程中实时跟踪遗忘"""
    
    def __init__(self):
        self.evaluator = ForgettingEvaluator()
        self.current_task_id = 0
        self.task_performance_matrix = {}  # 存储每个任务在每个阶段的性能
        self.all_task_names = []  # 所有任务名称
        
    def update_task_performance(self, task_name: str, stage_name: str, accuracy: float):
        """更新任务性能"""
        if task_name not in self.all_task_names:
            self.all_task_names.append(task_name)
        
        self.evaluator.record_task_performance_by_name(task_name, stage_name, accuracy)
    
    def evaluate_all_tasks(self, evaluation_fn, tasks_data, device):
        """评估所有已学习任务的性能"""
        task_accuracies = {}
        
        for task_id, (task_name, task_data) in enumerate(zip(self.all_task_names, tasks_data)):
            if task_id <= self.current_task_id:  # 只评估已学习的任务
                accuracy = evaluation_fn(task_data, device)
                task_accuracies[task_name] = accuracy
                
                # 记录性能到评估器
                stage_name = f"after_task_{self.current_task_id}"
                self.update_task_performance(task_name, stage_name, accuracy)
        
        self.current_task_id += 1
        return task_accuracies
    
    def get_current_forgetting_status(self) -> Dict:
        """获取当前遗忘状态"""
        return self.evaluator.get_comprehensive_metrics(self.all_task_names)
    
    def get_forgetting_evaluator(self) -> ForgettingEvaluator:
        """获取评估器实例"""
        return self.evaluator


# 示例使用
if __name__ == "__main__":
    # 创建示例数据来演示遗忘评估
    evaluator = ForgettingEvaluator()
    
    # 模拟任务性能历史 (任务名称, 阶段名称, 准确率)
    # 任务1在学习任务2后性能下降，显示遗忘
    evaluator.record_task_performance_by_name("task1", "after_task1", 0.85)
    evaluator.record_task_performance_by_name("task1", "after_task2", 0.75)  # 性能下降
    evaluator.record_task_performance_by_name("task1", "after_task3", 0.70)  # 进一步下降
    
    # 任务2在学习任务3后性能下降
    evaluator.record_task_performance_by_name("task2", "after_task2", 0.80)
    evaluator.record_task_performance_by_name("task2", "after_task3", 0.72)  # 性能下降
    
    # 任务3 (最新任务)
    evaluator.record_task_performance_by_name("task3", "after_task3", 0.88)
    
    # 计算指标
    metrics = evaluator.get_comprehensive_metrics(["task1", "task2", "task3"])
    
    print("综合评估指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # 详细分析
    evaluator.print_detailed_analysis()