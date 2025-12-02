"""
可视化工具 - 生成实验对比图表
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class Visualizer:
    """实验结果可视化工具"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_accuracy_comparison(self, results: Dict[str, Dict[str, Any]], filename: str = 'accuracy_comparison.png'):
        """绘制准确率对比图"""
        exp_names = list(results.keys())
        accuracies = []
        
        for exp_name in exp_names:
            exp_results = results[exp_name]
            if 'tasks' in exp_results:
                task_accs = []
                for task_results in exp_results['tasks'].values():
                    for metrics in task_results.values():
                        if 'accuracy' in metrics:
                            task_accs.append(metrics['accuracy'])
                accuracies.append(np.mean(task_accs) if task_accs else 0)
            else:
                accuracies.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.bar(exp_names, accuracies)
        plt.xlabel('实验配置')
        plt.ylabel('平均准确率')
        plt.title('消融实验准确率对比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
    
    def plot_forgetting_rate_comparison(self, results: Dict[str, Dict[str, Any]], filename: str = 'forgetting_rate_comparison.png'):
        """绘制遗忘率对比图"""
        exp_names = list(results.keys())
        forgetting_rates = [0.0] * len(exp_names)  # 简化实现
        
        plt.figure(figsize=(10, 6))
        plt.bar(exp_names, forgetting_rates)
        plt.xlabel('实验配置')
        plt.ylabel('平均遗忘率')
        plt.title('消融实验遗忘率对比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
    
    def plot_training_time_comparison(self, results: Dict[str, Dict[str, Any]], filename: str = 'training_time_comparison.png'):
        """绘制训练时间对比图"""
        exp_names = list(results.keys())
        training_times = [1.0] * len(exp_names)  # 简化实现
        
        plt.figure(figsize=(10, 6))
        plt.bar(exp_names, training_times)
        plt.xlabel('实验配置')
        plt.ylabel('训练时间 (小时)')
        plt.title('消融实验训练时间对比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300)
        plt.close()
    
    def generate_all_plots(self, results: Dict[str, Dict[str, Any]]):
        """生成所有对比图表"""
        self.plot_accuracy_comparison(results)
        self.plot_forgetting_rate_comparison(results)
        self.plot_training_time_comparison(results)
        print(f"所有图表已保存到: {self.output_dir}")