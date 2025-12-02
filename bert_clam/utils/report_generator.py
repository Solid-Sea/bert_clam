"""
报告生成器 - 自动生成Markdown格式的实验报告
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json


class ReportGenerator:
    """实验报告生成器"""
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
    
    def generate_report(self, results: Dict[str, Any], output_file: str = 'EXPERIMENT_REPORT.md'):
        """生成完整的实验报告"""
        report_lines = []
        
        # 标题
        report_lines.append("# BERT-CLAM 消融实验报告")
        report_lines.append("")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 实验概览
        report_lines.append("## 实验概览")
        report_lines.append("")
        if 'metadata' in results:
            metadata = results['metadata']
            report_lines.append(f"- **开始时间**: {metadata.get('start_time', 'N/A')}")
            report_lines.append(f"- **结束时间**: {metadata.get('end_time', 'N/A')}")
            report_lines.append(f"- **实验数量**: {len(metadata.get('experiments', []))}")
        report_lines.append("")
        
        # 实验配置列表
        report_lines.append("## 实验配置")
        report_lines.append("")
        if 'experiments' in results:
            for exp_name in results['experiments'].keys():
                report_lines.append(f"- {exp_name}")
        report_lines.append("")
        
        # 性能对比表
        report_lines.append("## 性能对比")
        report_lines.append("")
        report_lines.append("| 配置 | 平均准确率 | 平均F1 | Matthews相关系数 |")
        report_lines.append("|------|-----------|--------|-----------------|")
        
        if 'experiments' in results:
            for exp_name, exp_results in results['experiments'].items():
                avg_acc = self._compute_avg_metric(exp_results, 'accuracy')
                avg_f1 = self._compute_avg_metric(exp_results, 'f1')
                avg_mcc = self._compute_avg_metric(exp_results, 'matthews_correlation')
                report_lines.append(f"| {exp_name} | {avg_acc:.4f} | {avg_f1:.4f} | {avg_mcc:.4f} |")
        report_lines.append("")
        
        # 详细结果
        report_lines.append("## 详细结果")
        report_lines.append("")
        if 'experiments' in results:
            for exp_name, exp_results in results['experiments'].items():
                report_lines.append(f"### {exp_name}")
                report_lines.append("")
                if 'tasks' in exp_results:
                    for task_name, task_results in exp_results['tasks'].items():
                        report_lines.append(f"#### 任务: {task_name}")
                        report_lines.append("")
                        for eval_task, metrics in task_results.items():
                            report_lines.append(f"- **{eval_task}**:")
                            for metric_name, value in metrics.items():
                                if isinstance(value, float):
                                    report_lines.append(f"  - {metric_name}: {value:.4f}")
                        report_lines.append("")
        
        # 结论
        report_lines.append("## 结论")
        report_lines.append("")
        report_lines.append("本报告展示了BERT-CLAM模型在不同配置下的消融实验结果。")
        report_lines.append("")
        
        # 保存报告
        report_path = self.run_dir / output_file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"实验报告已生成: {report_path}")
        return report_path
    
    def _compute_avg_metric(self, exp_results: Dict[str, Any], metric_name: str) -> float:
        """计算平均指标"""
        values = []
        if 'tasks' in exp_results:
            for task_results in exp_results['tasks'].values():
                for metrics in task_results.values():
                    if metric_name in metrics:
                        values.append(metrics[metric_name])
        return sum(values) / len(values) if values else 0.0