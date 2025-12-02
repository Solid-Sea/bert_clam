"""
BERT-CLAM配置加载器
"""

import yaml
import json
from typing import Dict, Any, Union
import os
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径 (支持.yaml, .yml, .json)
    
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    if file_ext in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif file_ext == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {file_ext}")
    
    return config if config is not None else {}


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置，override_config会覆盖base_config中的相同键
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
    
    Returns:
        合并后的配置
    """
    import copy
    merged = copy.deepcopy(base_config)
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v
    
    deep_update(merged, override_config)
    return merged


def save_config(config: Dict[str, Any], output_path: str):
    """保存配置到文件
    
    Args:
        config: 配置字典
        output_path: 输出路径
    """
    file_ext = os.path.splitext(output_path)[1].lower()
    
    if file_ext in ['.yaml', '.yml']:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    elif file_ext == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的配置文件格式: {file_ext}")


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'model': {
            'model_name': 'bert-base-uncased',
            'num_labels': 2,
            'hidden_size': 768,
            'num_attention_heads': 12
        },
        'lora': {
            'r': 8,
            'alpha': 16,
            'dropout': 0.1,
            'target_modules': ['query', 'key', 'value', 'classifier']
        },
        'training': {
            'batch_size': 64,
            'learning_rate': 2e-5,
            'num_epochs': 3,
            'warmup_steps': 100,
            'gradient_clip': 1.0
        },
        'amr': {
            'k': 15,
            'memory_dim': 768,
            'faiss_index_type': 'IndexFlatL2'
        },
        'ewc': {
            'lambda_ewc': 0.15,
            'fisher_samples': 100
        },
        'alp': {
            'top_k': 3,
            'similarity_threshold': 0.7,
            'adaptive_scaling': True,
            'task_similarity_aware': True
        },
        'grammar_aware': {
            'grammar_features_dim': 64,  # 确保不为0
            'max_seq_length': 128  # 减少序列长度以节省内存
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置的完整性
    
    Args:
        config: 配置字典
    
    Returns:
        配置是否有效
    """
    required_keys = ['model', 'training']
    
    for key in required_keys:
        if key not in config:
            print(f"警告: 配置中缺少必需的键: {key}")
            return False
    
    # 验证模型配置
    model_config = config.get('model', {})
    required_model_keys = ['model_name', 'num_labels']
    for key in required_model_keys:
        if key not in model_config:
            print(f"警告: 模型配置中缺少必需的键: {key}")
            return False
    
    # 验证训练配置
    training_config = config.get('training', {})
    required_training_keys = ['batch_size', 'learning_rate', 'num_epochs']
    for key in required_training_keys:
        if key not in training_config:
            print(f"警告: 训练配置中缺少必需的键: {key}")
            return False
    
    return True


def load_experiment_config(experiment_name: str = 'prototype') -> Dict[str, Any]:
    """加载实验特定配置"""
    base_config = get_default_config()
    
    # 根据实验类型调整配置
    if experiment_name == 'prototype':
        # 快速原型验证
        experiment_config = {
            'training': {
                'batch_size': 32,
                'num_epochs': 2,
                'learning_rate': 2e-5
            },
            'experiments': {
                'n_tasks': 3,
                'tasks': ['cola', 'sst2', 'mrpc']
            }
        }
    elif experiment_name == 'ablation':
        # 消融实验
        experiment_config = {
            'training': {
                'batch_size': 64,
                'num_epochs': 3,
                'learning_rate': 2e-5
            },
            'experiments': {
                'n_tasks': 5,
                'tasks': ['cola', 'sst2', 'mrpc', 'qnli', 'qqp'],
                'n_runs': 5,
                'seeds': [42, 123, 456, 789, 1024]
            }
        }
    elif experiment_name == 'benchmark':
        # 基准对比
        experiment_config = {
            'training': {
                'batch_size': 64,
                'num_epochs': 3,
                'learning_rate': 2e-5
            },
            'experiments': {
                'n_tasks': 10,
                'tasks': ['cola', 'sst2', 'mrpc', 'qnli', 'qqp', 'rte', 'wnli', 'mnli', 'stsb', 'ax'],
                'n_runs': 5
            }
        }
    else:
        experiment_config = {}
    
    return merge_configs(base_config, experiment_config)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = get_default_config()
        
        self.original_config = self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        self.config = merge_configs(self.config, updates)
    
    def get(self, key_path: str, default=None):
        """获取配置值，支持嵌套键路径 (如 'model.hidden_size')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """设置配置值，支持嵌套键路径"""
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def save(self, output_path: str):
        """保存当前配置"""
        save_config(self.config, output_path)
    
    def reset(self):
        """重置为原始配置"""
        self.config = self.original_config.copy()


def load_ablation_configs(ablation_config_path: str) -> Dict[str, Dict[str, Any]]:
    """加载消融实验配置
    
    Args:
        ablation_config_path: 消融配置文件路径
    
    Returns:
        实验配置字典，键为实验名称，值为完整配置
    """
    ablation_config = load_config(ablation_config_path)
    base_config = ablation_config.get('base_config', {})
    experiments = ablation_config.get('experiments', {})
    
    experiment_configs = {}
    for exp_name, exp_overrides in experiments.items():
        experiment_configs[exp_name] = merge_configs(base_config, exp_overrides)
    
    return experiment_configs


def get_experiment_config(ablation_config_path: str, experiment_name: str) -> Dict[str, Any]:
    """获取特定消融实验的配置
    
    Args:
        ablation_config_path: 消融配置文件路径
        experiment_name: 实验名称
    
    Returns:
        实验配置字典
    """
    experiment_configs = load_ablation_configs(ablation_config_path)
    
    if experiment_name not in experiment_configs:
        raise ValueError(f"实验配置不存在: {experiment_name}")
    
    return experiment_configs[experiment_name]