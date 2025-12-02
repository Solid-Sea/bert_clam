"""
LoRA适配器模块
参数高效的微调方法
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import math


class LoRAAdapter(nn.Module):
    """LoRA适配器实现"""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 r: int = 8,
                 alpha: int = 16,
                 dropout: float = 0.0,
                 bias: bool = False):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            r: LoRA秩
            alpha: 缩放因子
            dropout: dropout概率
            bias: 是否使用偏置
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        
        # LoRA参数
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * math.sqrt(1 / r))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        # 标记是否激活LoRA
        self.active = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if not self.active:
            # 如果未激活，只应用偏置
            result = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
            if self.bias is not None:
                result = result + self.bias
            return result
        
        # 应用dropout
        x_dropout = self.dropout(x)
        
        # LoRA变换: x -> x @ A.T -> (x @ A.T) @ B.T
        # 等价于: x @ (A.T @ B.T) = x @ (B @ A).T
        lora_A_weight = self.lora_A
        lora_B_weight = self.lora_B
        
        # 计算LoRA输出
        result = x_dropout @ lora_A_weight.T @ lora_B_weight.T
        result = result * self.scaling
        
        if self.bias is not None:
            result = result + self.bias
            
        return result
    
    def get_delta_weight(self) -> torch.Tensor:
        """获取LoRA增量权重 B @ A * scaling"""
        return self.lora_B @ self.lora_A * self.scaling
    
    def merge_weights(self) -> torch.Tensor:
        """合并权重（如果需要永久合并）"""
        return self.get_delta_weight()
    
    def activate(self):
        """激活LoRA"""
        self.active = True
    
    def deactivate(self):
        """停用LoRA"""
        self.active = False


class MultiTaskLoRA(nn.Module):
    """多任务LoRA适配器"""
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 8,
                 alpha: int = 16,
                 dropout: float = 0.0,
                 max_tasks: int = 10):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.max_tasks = max_tasks
        
        # 为每个任务存储LoRA参数
        self.task_loras = nn.ModuleDict()
        self.task_count = 0
        
        # 默认LoRA（任务0）
        default_lora = LoRAAdapter(in_features, out_features, r, alpha, dropout)
        self.task_loras['0'] = default_lora
    
    def add_task(self, task_id: int) -> bool:
        """为新任务添加LoRA适配器"""
        if task_id >= self.max_tasks:
            return False
        
        if str(task_id) not in self.task_loras:
            lora = LoRAAdapter(
                self.in_features,
                self.out_features,
                self.r,
                self.alpha,
                0.0  # 不对特定任务使用dropout
            )
            # 将新的LoRA模块移动到正确的设备
            lora.to(next(self.parameters()).device)
            self.task_loras[str(task_id)] = lora
            self.task_count += 1
            return True
        
        return False
    
    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """前向传播"""
        task_key = str(task_id)
        if task_key in self.task_loras:
            return self.task_loras[task_key](x)
        else:
            # 如果任务不存在，使用默认任务0
            return self.task_loras['0'](x)
    
    def get_task_embedding(self, task_id: int) -> torch.Tensor:
        """获取任务嵌入（基于LoRA参数）"""
        task_key = str(task_id)
        if task_key in self.task_loras:
            lora = self.task_loras[task_key]
            # 将LoRA参数展平并连接作为任务嵌入
            lora_params = torch.cat([
                lora.lora_A.flatten(),
                lora.lora_B.flatten()
            ])
            return lora_params
        else:
            return self.get_task_embedding(0)  # 返回默认任务嵌入


class LoRAInjectedLinear(nn.Module):
    """结合了原始Linear层和MultiTaskLoRA的注入层"""
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int, dropout: float, max_tasks: int):
        super().__init__()
        self.linear = original_linear
        self.lora = MultiTaskLoRA(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            max_tasks=max_tasks
        )
        self.current_task_id = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始线性变换
        original_output = self.linear(x)
        # LoRA增量变换
        lora_output = self.lora(x, self.current_task_id)
        return original_output + lora_output

    def set_task(self, task_id: int):
        self.current_task_id = task_id

    def add_task(self, task_id: int):
        return self.lora.add_task(task_id)

class BERTLoRAModifier:
    """BERT模型的LoRA修改器 (使用猴子补丁)"""
    
    def __init__(self,
                 base_model: nn.Module,
                 target_modules: List[str] = ['query', 'key', 'value', 'dense'],
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 max_tasks: int = 10):
        
        self.base_model = base_model
        self.target_modules = target_modules
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_tasks = max_tasks
        
        self.injected_modules = {}
        
        # 冻结原始模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 为BERT的指定模块添加LoRA
        self._inject_lora_to_bert()

    def _inject_lora_to_bert(self):
        """使用猴子补丁为BERT模型的指定模块注入LoRA"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # 创建注入层
                    injected_layer = LoRAInjectedLinear(
                        original_linear=module,
                        r=self.lora_r,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                        max_tasks=self.max_tasks
                    )
                    
                    # 获取父模块和属性名
                    parent_name, child_name = name.rsplit('.', 1)
                    parent_module = self.base_model.get_submodule(parent_name)
                    
                    # 替换原始模块
                    setattr(parent_module, child_name, injected_layer)
                    self.injected_modules[name] = injected_layer

    def set_task(self, task_id: int):
        """为所有注入的模块设置当前任务ID"""
        for module in self.injected_modules.values():
            module.set_task(task_id)

    def add_task(self, task_id: int) -> bool:
        """为新任务添加LoRA适配器"""
        success = True
        for injected_layer in self.injected_modules.values():
            if not injected_layer.add_task(task_id):
                success = False
        return success
    
    def get_task_embedding(self, task_id: int, module_name: str = None) -> torch.Tensor:
        """获取任务嵌入"""
        if module_name:
            # 将模块名转换为安全名称
            safe_module_name = module_name.replace('.', '_')
            if safe_module_name in self.lora_adapters:
                return self.lora_adapters[safe_module_name].get_task_embedding(task_id)
        elif len(self.lora_adapters) > 0:
            # 返回第一个适配器的任务嵌入
            first_adapter = next(iter(self.lora_adapters.values()))
            return first_adapter.get_task_embedding(task_id)
        else:
            # 创建随机任务嵌入
            return torch.randn(self.lora_r * 2)  # 简化的任务嵌入


class LoRAManager:
    """LoRA管理器"""
    
    def __init__(self):
        self.active_adapters = {}
        self.task_registry = {}
    
    def register_adapter(self, name: str, adapter: LoRAAdapter):
        """注册LoRA适配器"""
        self.active_adapters[name] = adapter
    
    def register_task(self, task_id: int, adapter_names: List[str]):
        """注册任务和相关的适配器"""
        self.task_registry[task_id] = adapter_names
    
    def activate_task(self, task_id: int):
        """激活特定任务的适配器"""
        if task_id in self.task_registry:
            for adapter_name in self.task_registry[task_id]:
                if adapter_name in self.active_adapters:
                    self.active_adapters[adapter_name].activate()
    
    def deactivate_all(self):
        """停用所有适配器"""
        for adapter in self.active_adapters.values():
            adapter.deactivate()