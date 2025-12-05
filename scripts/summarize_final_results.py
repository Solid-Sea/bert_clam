#!/usr/bin/env python3
"""
Final Results Summary Script

This script analyzes all experiment results and generates a Markdown table
sorted by average forgetting rate in ascending order.
"""

import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set font to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Handle minus signs correctly

def read_experiment_results(experiments_dir):
    """Read training history from all experiment directories."""
    results = []
    
    # Find all training_history.json files in subdirectories
    history_files = glob.glob(os.path.join(experiments_dir, "**", "training_history.json"), recursive=True)
    
    for file_path in history_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract experiment name from path
            experiment_path = Path(file_path)
            experiment_name = experiment_path.parent.name
            experiment_group = experiment_path.parent.parent.name
            
            # Get metrics
            avg_forgetting = data.get('average_forgetting', 0.0)
            final_accuracies = data.get('final_accuracies', {})
            
            # Calculate average accuracy
            if final_accuracies:
                avg_accuracy = sum(final_accuracies.values()) / len(final_accuracies)
            else:
                avg_accuracy = 0.0
            
            results.append({
                'experiment_group': experiment_group,
                'experiment_name': experiment_name,
                'average_forgetting': avg_forgetting,
                'final_average_accuracy': avg_accuracy,
                'final_accuracies': final_accuracies
            })
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return results

def generate_markdown_table(results):
    """Generate a Markdown table sorted by average forgetting rate."""
    # Sort by average forgetting rate (ascending)
    sorted_results = sorted(results, key=lambda x: x['average_forgetting'])
    
    # Create Markdown table
    table = "| 实验名称 | 实验组 | 平均遗忘率 | 最终平均准确率 | 最终任务准确率 |\n"
    table += "|---------|--------|-----------|--------------|-------------|\n"
    
    for result in sorted_results:
        experiment_name = result['experiment_name']
        experiment_group = result['experiment_group']
        avg_forgetting = f"{result['average_forgetting']:.3f}"
        avg_accuracy = f"{result['final_average_accuracy']:.3f}"
        final_accuracies = str(result['final_accuracies'])
        
        table += f"| {experiment_name} | {experiment_group} | {avg_forgetting} | {avg_accuracy} | {final_accuracies} |\n"
    
    return table, sorted_results

def create_forgetting_comparison_chart(sorted_results):
    """Create a bar chart comparing forgetting rates across experiments."""
    # Prepare data
    experiment_names = [result['experiment_name'] for result in sorted_results]
    forgetting_rates = [result['average_forgetting'] for result in sorted_results]
    
    # Define colors based on forgetting rate ranges
    colors = []
    for rate in forgetting_rates:
        if rate == 0.0:
            colors.append('#2E8B57')  # Sea Green for zero forgetting
        elif rate < 0.1:
            colors.append('#32CD32')  # Lime Green for low forgetting
        elif rate < 0.2:
            colors.append('#FFD700')  # Gold for medium forgetting
        elif rate < 0.3:
            colors.append('#FF8C00')  # Dark Orange for high forgetting
        else:
            colors.append('#DC143C')  # Crimson for very high forgetting
    
    # Create the bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(experiment_names)), forgetting_rates, color=colors)
    
    # Customize the chart
    plt.title('实验遗忘率对比 (按遗忘率升序排列)', fontsize=16, fontweight='bold')
    plt.xlabel('实验配置', fontsize=12)
    plt.ylabel('平均遗忘率', fontsize=12)
    plt.xticks(range(len(experiment_names)), experiment_names, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, (bar, rate) in enumerate(zip(bars, forgetting_rates)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart
    chart_path = 'final_forgetting_comparison.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"遗忘率对比图表已保存至: {chart_path}")
    return chart_path

def main():
    """Main function to generate summary and chart."""
    experiments_dir = 'experiments'
    
    if not os.path.exists(experiments_dir):
        print(f"Error: Experiments directory '{experiments_dir}' does not exist.")
        return
    
    print("Reading experiment results...")
    results = read_experiment_results(experiments_dir)
    
    if not results:
        print("No experiment results found.")
        return
    
    print(f"Found {len(results)} experiment results.")
    
    print("Generating Markdown table...")
    markdown_table, sorted_results = generate_markdown_table(results)
    
    print("\n" + "="*50)
    print("MARKDOWN TABLE (sorted by average forgetting rate):")
    print("="*50)
    print(markdown_table)
    
    print("Creating forgetting rate comparison chart...")
    chart_path = create_forgetting_comparison_chart(sorted_results)
    
    print(f"\nChart saved to: {chart_path}")
    print("\nData processing complete!")

if __name__ == "__main__":
    main()