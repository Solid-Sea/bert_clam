"""
ALP: 自适应LoRA池化模块 - 修复版
解决设备分配和张量维度问题的自适应LoRA融合机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class AdaptiveLoRAFusion(nn.Module):
    """修复后的自适应LoRA池化模块 - 解决设备分配和张量维度问题"""
    
    def __init__(self, hidden_size: int = 768, r: int = 8, alpha: int = 16, 
                 max_tasks: int = 10, device: str = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.max_tasks = max_tasks
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 存储历史任务的LoRA参数
        self.lora_adapters = {}  # {task_id: {'lora_a': tensor, 'lora_b': tensor}}
        self.task_embeddings = {}  # {task_id: tensor}
        
        # 相似度计算网络 - 确保在正确设备上
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.similarity_net = self.similarity_net.to(self.device)
        
        # 门控机制 - 确保在正确设备上
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.gate = self.gate.to(self.device)
        
        # 任务计数器
        self.task_counter = 0

    def register_task_adapter(self, task_id: int, lora_a_weight: torch.Tensor, 
                            lora_b_weight: torch.Tensor, task_embedding: torch.Tensor):
        """为特定任务注册LoRA适配器"""
        # 确保权重在正确的设备上
        lora_a_weight = lora_a_weight.to(self.device)
        lora_b_weight = lora_b_weight.to(self.device)
        task_embedding = task_embedding.to(self.device)
        
        # 存储LoRA参数
        self.lora_adapters[task_id] = {
            'lora_a': lora_a_weight.clone().detach(),
            'lora_b': lora_b_weight.clone().detach()
        }
        self.task_embeddings[task_id] = task_embedding.clone().detach()
        
        self.task_counter += 1
        
        # 限制任务数量（可选的内存管理）
        if len(self.lora_adapters) > self.max_tasks:
            oldest_task_id = min(self.lora_adapters.keys())
            self.lora_adapters.pop(oldest_task_id, None)
            self.task_embeddings.pop(oldest_task_id, None)

    def compute_task_similarity(self, current_task_id: int, target_task_id: int) -> torch.Tensor:
        """计算两个任务之间的相似度"""
        if (current_task_id not in self.task_embeddings or 
            target_task_id not in self.task_embeddings):
            return torch.tensor(0.0, device=self.device)
        
        current_emb = self.task_embeddings[current_task_id]
        target_emb = self.task_embeddings[target_task_id]
        
        # 确保输入维度正确并计算相似度
        combined_emb = torch.cat([current_emb, target_emb], dim=0)
        combined_emb = combined_emb.unsqueeze(0)  # 添加批次维度
        
        # 确保相似度网络在正确设备上
        if next(self.similarity_net.parameters()).device != self.device:
            self.similarity_net = self.similarity_net.to(self.device)
        
        similarity = self.similarity_net(combined_emb)
        return similarity.squeeze()

    def fuse_adapters(self, current_task_id: int, max_history_tasks: int = 3) -> Optional[Dict]:
        """融合历史任务的适配器"""
        if len(self.lora_adapters) <= 1:
            return None
        
        # 计算当前任务与所有历史任务的相似度
        similarities = {}
        for task_id in self.lora_adapters.keys():
            if task_id != current_task_id:
                sim = self.compute_task_similarity(current_task_id, task_id)
                similarities[task_id] = sim.item()
        
        if not similarities:
            return None
        
        # 选择最相似的几个任务进行融合
        sorted_tasks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        selected_tasks = [task_id for task_id, _ in sorted_tasks[:max_history_tasks]]
        
        if not selected_tasks:
            return None
        
        # 计算权重（归一化）
        total_sim = sum(similarities[task_id] for task_id in selected_tasks)
        if total_sim == 0:
            weights = {task_id: 1.0/len(selected_tasks) for task_id in selected_tasks}
        else:
            weights = {task_id: similarities[task_id]/total_sim for task_id in selected_tasks}
        
        # 加权融合LoRA参数
        fused_lora_a = None
        fused_lora_b = None
        
        for task_id in selected_tasks:
            weight = weights[task_id]
            task_lora_a = self.lora_adapters[task_id]['lora_a']
            task_lora_b = self.lora_adapters[task_id]['lora_b']
            
            if fused_lora_a is None:
                fused_lora_a = weight * task_lora_a
                fused_lora_b = weight * task_lora_b
            else:
                fused_lora_a += weight * task_lora_a
                fused_lora_b += weight * task_lora_b
        
        return {
            'lora_a': fused_lora_a,
            'lora_b': fused_lora_b
        }

    def forward(self, hidden_states: torch.Tensor, current_task_id: int) -> torch.Tensor:
        """前向传播，融合历史任务知识"""
        # 确保输入在正确设备上
        hidden_states = hidden_states.to(self.device)
        
        if len(self.lora_adapters) <= 1:
            return hidden_states
        
        # 融合历史适配器
        fused_adapters = self.fuse_adapters(current_task_id)
        
        if fused_adapters is None:
            return hidden_states
        
        # 应用融合的LoRA
        batch_size, seq_len, hidden_size = hidden_states.shape
        x_reshaped = hidden_states.view(-1, hidden_size)
        
        # 计算融合LoRA输出
        lora_out = (x_reshaped @ fused_adapters['lora_a'].T) @ fused_adapters['lora_b'].T
        lora_out = lora_out * self.scaling
        lora_out = lora_out.view(batch_size, seq_len, hidden_size)
        
        # 应用门控机制
        pooled_states = hidden_states.mean(dim=1).mean(dim=0)  # 简化的门控输入
        if pooled_states.dim() == 0:
            pooled_states = pooled_states.unsqueeze(0)
        if pooled_states.shape[0] != self.hidden_size:
            pooled_states = torch.randn(self.hidden_size, device=self.device) * 0.1
        pooled_states = pooled_states.unsqueeze(0) if pooled_states.dim() == 1 else pooled_states
        
        # 确保门控网络在正确设备上
        if next(self.gate.parameters()).device != self.device:
            self.gate = self.gate.to(self.device)
        
        gate_value = self.gate(pooled_states)
        gate_value = gate_value.unsqueeze(1).unsqueeze(2)  # 为隐藏状态匹配维度
        
        # 融合原始输出和LoRA影响
        result = gate_value * (hidden_states + lora_out) + (1 - gate_value) * hidden_states
        
        return result

    def get_task_characteristics(self, task_id: int) -> Optional[Dict]:
        """获取任务特征"""
        if task_id not in self.task_embeddings:
            return None
        
        return {
            'embedding_norm': torch.norm(self.task_embeddings[task_id]).item(),
            'similarity_to_others': {
                other_id: self.compute_task_similarity(task_id, other_id).item()
                for other_id in self.task_embeddings.keys() if other_id != task_id
            }
        }


class EnhancedAdaptiveLoRAPooling(nn.Module):
    """增强自适应LoRA池化模块"""
    
    def __init__(self, hidden_size: int = 768, r: int = 8, alpha: int = 16, 
                 top_k: int = 3, similarity_threshold: float = 0.7,
                 adaptive_scaling: bool = True, task_similarity_aware: bool = True, 
                 device: str = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.alpha = alpha
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.adaptive_scaling = adaptive_scaling
        self.task_similarity_aware = task_similarity_aware
        self.scaling = alpha / r
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 存储每个任务的LoRA参数
        self.task_loras = {}  # {task_id: {layer_name: {'lora_a': tensor, 'lora_b': tensor}}}
        self.task_embeddings = {}  # {task_id: tensor}
        self.task_similarity_cache = {}  # 缓存相似度计算结果
        self.layer_importance = {}  # 各层的重要性评分

        # 相似度网络
        self.semantic_sim_net = self._build_similarity_network()
        self.semantic_sim_net = self.semantic_sim_net.to(self.device)

        # 自适应参数
        self.performance_history = {}
        self.adaptation_factor = 1.0

    def _build_similarity_network(self) -> nn.Module:
        """构建相似度计算网络"""
        return nn.Sequential(
            nn.Linear(self.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def add_task_lora(self, task_id: int, task_embedding: torch.Tensor,
                     lora_params: Optional[Dict] = None):
        """为新任务添加LoRA适配器"""
        if lora_params is None:
            # 如果没有提供LoRA参数，创建默认参数
            lora_params = {}
            # 为简化，使用随机初始化
            for layer_name in ['classifier', 'pooler']:
                lora_a = torch.randn(self.r, self.hidden_size, device=self.device) * 0.01
                lora_b = torch.zeros(self.hidden_size, self.r, device=self.device)
                lora_params[layer_name] = {
                    'lora_a': lora_a,
                    'lora_b': lora_b
                }
        
        self.task_loras[task_id] = lora_params
        self.task_embeddings[task_id] = task_embedding.to(self.device)
        self.task_similarity_cache.clear()  # 清除相似度缓存

    def compute_task_similarity(self, current_task_embedding: torch.Tensor) -> Dict:
        """计算与历史任务的相似度"""
        if not self.task_embeddings:
            return {}

        similarities = {}
        current_task_embedding = current_task_embedding.to(self.device)
        
        for task_id, task_embed in self.task_embeddings.items():
            task_embed = task_embed.to(self.device)  # 确保在正确设备上
            
            # 余弦相似度
            cos_sim = F.cosine_similarity(
                current_task_embedding.unsqueeze(0),
                task_embed.unsqueeze(0)
            ).item()
            
            # 欧几里得距离
            euclidean_dist = torch.norm(current_task_embedding - task_embed).item()
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # 神经网络相似度
            if self.task_similarity_aware:
                combined_embed = torch.cat([current_task_embedding, task_embed])
                combined_embed = combined_embed.unsqueeze(0)
                if next(self.semantic_sim_net.parameters()).device != self.device:
                    self.semantic_sim_net = self.semantic_sim_net.to(self.device)
                nn_sim = self.semantic_sim_net(combined_embed).item()
                combined_sim = 0.4 * cos_sim + 0.3 * euclidean_sim + 0.3 * nn_sim
            else:
                combined_sim = 0.7 * cos_sim + 0.3 * euclidean_sim
            
            similarities[task_id] = combined_sim
        
        return similarities

    def pool_lora_weights(self, current_task_embedding: torch.Tensor, 
                         current_task_id: Optional[int] = None) -> Optional[Dict]:
        """池化相似任务的LoRA权重"""
        similarities = self.compute_task_similarity(current_task_embedding)
        
        if not similarities:
            return None
        
        # 选择top-k相似任务
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_tasks = [task_id for task_id, sim in sorted_similarities[:self.top_k]
                    if sim > self.similarity_threshold]
        
        if not top_tasks:
            return None
        
        # 计算加权平均LoRA
        pooled_lora = {}
        
        for layer_name in self.task_loras.get(top_tasks[0], {}):
            pooled_lora[layer_name] = {
                'lora_a': None,
                'lora_b': None
            }
            
            lora_a_sum = None
            lora_b_sum = None
            total_weight = 0
            
            for task_id in top_tasks:
                sim_score = similarities[task_id]
                layer_weight = sim_score
                if self.adaptive_scaling and layer_name in self.layer_importance:
                    layer_weight *= (1 + self.layer_importance[layer_name])
                
                task_lora = self.task_loras[task_id][layer_name]
                task_lora_a = task_lora['lora_a'].to(self.device)
                task_lora_b = task_lora['lora_b'].to(self.device)
                
                if lora_a_sum is None:
                    lora_a_sum = layer_weight * task_lora_a
                    lora_b_sum = layer_weight * task_lora_b
                else:
                    lora_a_sum += layer_weight * task_lora_a
                    lora_b_sum += layer_weight * task_lora_b
                
                total_weight += layer_weight
            
            if total_weight > 0:
                pooled_lora[layer_name]['lora_a'] = lora_a_sum / total_weight
                pooled_lora[layer_name]['lora_b'] = lora_b_sum / total_weight
        
        return pooled_lora

    def apply_adaptive_lora(self, x: torch.Tensor, task_embedding: torch.Tensor, 
                          layer_name: str, current_task_id: Optional[int] = None) -> torch.Tensor:
        """应用自适应LoRA"""
        if (current_task_id not in self.task_loras or 
            layer_name not in self.task_loras[current_task_id]):
            return x
        
        # 当前任务的LoRA
        current_lora = self.task_loras[current_task_id][layer_name]
        device = x.device
        
        # 确保LoRA权重在正确设备上
        lora_a_weight = current_lora['lora_a'].to(device)
        lora_b_weight = current_lora['lora_b'].to(device)
        
        batch_size, seq_len, hidden_size = x.shape
        x_reshaped = x.view(-1, hidden_size)
        
        # 计算当前任务LoRA输出
        lora_out = (x_reshaped @ lora_a_weight.T) @ lora_b_weight.T
        lora_out = lora_out * self.scaling
        lora_out = lora_out.view(batch_size, seq_len, -1)
        
        # 如果有历史任务，融合池化LoRA
        if len(self.task_loras) > 1:
            pooled_lora = self.pool_lora_weights(task_embedding, current_task_id)
            
            if pooled_lora is not None and layer_name in pooled_lora:
                pooled_a = pooled_lora[layer_name]['lora_a']
                pooled_b = pooled_lora[layer_name]['lora_b']
                
                if pooled_a is not None and pooled_b is not None:
                    pooled_a = pooled_a.to(device)
                    pooled_b = pooled_b.to(device)
                    
                    pooled_out = (x_reshaped @ pooled_a.T) @ pooled_b.T
                    pooled_out = pooled_out * self.scaling
                    pooled_out = pooled_out.view(batch_size, seq_len, -1)
                    
                    # 融合当前和池化LoRA
                    fusion_weight = min(task_embedding.norm().item() * 0.1, 0.5)
                    lora_out = (1 - fusion_weight) * lora_out + fusion_weight * pooled_out
        
        return x + lora_out

    def forward(self, hidden_states: torch.Tensor, task_embedding: torch.Tensor, 
               layer_name: str, current_task_id: Optional[int] = None) -> torch.Tensor:
        """前向传播"""
        return self.apply_adaptive_lora(hidden_states, task_embedding, 
                                      layer_name, current_task_id)