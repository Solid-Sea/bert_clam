"""
实验追踪器 - 收集、汇总和保存实验结果
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ExperimentTracker:
    """实验追踪器"""
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.results_dir = self.run_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'experiments': []
        }
    
    def log_experiment(self, exp_name: str, results: Dict[str, Any]):
        """记录单个实验结果"""
        self.all_results[exp_name] = results
        self.metadata['experiments'].append(exp_name)
        
        # 保存单个实验结果
        exp_result_file = self.results_dir / f'{exp_name}_results.json'
        with open(exp_result_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_summary(self):
        """生成汇总结果"""
        self.metadata['end_time'] = datetime.now().isoformat()
        
        # 汇总所有实验结果
        aggregated = {
            'metadata': self.metadata,
            'experiments': self.all_results,
            'summary': self._compute_summary()
        }
        
        # 保存汇总结果
        summary_file = self.results_dir / 'aggregated_results.json'
        with open(summary_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"汇总结果已保存到: {summary_file}")
    
    def _compute_summary(self) -> Dict[str, Any]:
        """计算汇总统计"""
        summary = {
            'num_experiments': len(self.all_results),
            'experiment_names': list(self.all_results.keys())
        }
        
        # 计算平均指标
        all_accuracies = []
        for exp_name, results in self.all_results.items():
            if 'tasks' in results:
                for task_name, task_results in results['tasks'].items():
                    for eval_task, metrics in task_results.items():
                        if 'accuracy' in metrics:
                            all_accuracies.append(metrics['accuracy'])
        
        if all_accuracies:
            summary['average_accuracy'] = sum(all_accuracies) / len(all_accuracies)
        
        return summary