"""
AMR: 自适应记忆检索模块 - 改进版ISDL
基于FAISS的知识存储和检索，用于持续学习中的知识保留
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List

# FAISS是可选依赖
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


class AdaptiveMemoryRetrieval:
    """自适应记忆检索 - 基于FAISS的知识存储和检索"""
    
    def __init__(self, dim: int = 768, k: int = 10, index_type: str = "IndexFlatL2"):
        """
        Args:
            dim: 特征维度 (BERT hidden size = 768)
            k: 检索的近邻数量
            index_type: FAISS索引类型
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required for AdaptiveMemoryRetrieval. "
                            "Please install it with: pip install faiss-cpu (or faiss-gpu for GPU support)")
        
        self.dim = dim
        self.k = k
        self.index_type = index_type
        
        # 初始化FAISS索引
        if index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dim)
        elif index_type == "IndexIVFFlat":
            # 需要先训练量化器
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, min(100, max(1, dim // 4)))
        else:
            self.index = faiss.IndexFlatL2(dim)
        
        self.memory_bank = {}  # 存储任务特定的记忆
        self.labels = []  # 存储标签
        self.vector_storage = []  # 存储实际向量用于检索
        self.count = 0
        self.is_trained = index_type != "IndexIVFFlat"  # IndexIVFFlat需要训练
        
    def add_memory(self, task_id: int, states: torch.Tensor, labels: torch.Tensor = None):
        """添加状态到记忆银行"""
        states_np = states.detach().cpu().numpy().astype('float32')
        
        # 处理多维tensor：取平均池化到正确维度
        if states_np.ndim > 2:
            # [batch, seq, dim] -> [batch, dim]
            states_np = states_np.mean(axis=1)
        elif states_np.ndim == 1:
            states_np = states_np.reshape(1, -1)
        
        # 确保维度匹配
        if states_np.shape[-1] != self.dim:
            return  # 跳过维度不匹配的数据
        
        # 训练IVF索引（如果需要）
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.is_trained:
            self.index.train(states_np)
            self.is_trained = True
        
        # 添加到索引
        self.index.add(states_np)
        
        # 存储实际向量
        for vec in states_np:
            self.vector_storage.append(vec)
        
        # 存储任务ID和标签信息
        if task_id not in self.memory_bank:
            self.memory_bank[task_id] = {
                'states': [],
                'labels': []
            }
        
        self.memory_bank[task_id]['states'].append(states_np)
        if labels is not None:
            self.memory_bank[task_id]['labels'].append(labels.cpu().numpy())
        
        self.count += len(states_np)
    
    def retrieve_knowledge(self, query: torch.Tensor, task_id: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索相关知识 - 真实的FAISS检索实现"""
        query_np = query.detach().cpu().numpy().astype('float32')
        
        if query_np.ndim > 2:
            query_np = query_np.mean(axis=1)
        elif query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        
        k = min(self.k, self.count)
        if k == 0:
            device = query.device
            return torch.zeros_like(query, device=device), torch.zeros(query.size(0), dtype=torch.long, device=device)
        
        try:
            distances, indices = self.index.search(query_np, k)
            
            # 从向量存储中检索实际的历史向量
            retrieved_vectors = []
            for idx_list in indices:
                valid_indices = idx_list[idx_list >= 0]
                if len(valid_indices) > 0:
                    first_idx = int(valid_indices[0])
                    if first_idx < len(self.vector_storage):
                        retrieved_vectors.append(self.vector_storage[first_idx])
                    else:
                        retrieved_vectors.append(query_np[0])
                else:
                    retrieved_vectors.append(query_np[0])
            
            retrieved_states = torch.from_numpy(np.array(retrieved_vectors)).to(query.device)
            distances_tensor = torch.from_numpy(distances).to(query.device)
            
            # 检索标签
            retrieved_labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
            if self.memory_bank and task_id in self.memory_bank:
                task_labels = self.memory_bank[task_id]['labels']
                if task_labels and len(task_labels) > 0:
                    labels_concat = np.concatenate(task_labels, axis=0) if len(task_labels) > 1 else task_labels[0]
                    for i, idx_list in enumerate(indices):
                        valid_indices = idx_list[idx_list >= 0]
                        if len(valid_indices) > 0 and valid_indices[0] < len(labels_concat):
                            retrieved_labels[i] = labels_concat[valid_indices[0]]
            
            return retrieved_states, retrieved_labels
        except Exception as e:
            device = query.device
            return torch.zeros_like(query, device=device), torch.zeros(query.size(0), dtype=torch.long, device=device)
    
    def compute_distillation_loss(self, current_states: torch.Tensor,
                                teacher_states: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
        """计算知识蒸馏损失 - KL散度"""
        student_log_probs = F.log_softmax(current_states / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_states / temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    def get_memory_size(self) -> int:
        """获取记忆库大小"""
        return self.count
    
    def clear_memory(self):
        """清空记忆库"""
        if hasattr(self.index, 'reset'):
            self.index.reset()
        self.memory_bank.clear()
        self.labels.clear()
        self.vector_storage.clear()
        self.count = 0
        self.is_trained = self.index_type != "IndexIVFFlat"


class EnhancedAdaptiveMemoryRetrieval(nn.Module):
    """增强版自适应记忆检索模块"""
    
    def __init__(self, hidden_size: int = 768, k: int = 10, memory_dim: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.memory_dim = memory_dim
        
        # 记忆检索组件
        self.amr_core = AdaptiveMemoryRetrieval(dim=memory_dim, k=k)
        
        # 相似度计算网络
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 知识融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """前向传播 - 真实的知识检索和融合"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 检索相关知识
        query = hidden_states.mean(dim=1)  # [batch, hidden_size]
        retrieved_states, retrieved_labels = self.amr_core.retrieve_knowledge(query, task_id)
        
        if retrieved_states.size(0) > 0 and self.amr_core.count > 0:
            # 计算融合权重
            fusion_weight = self.fusion_gate(query)  # [batch, 1]
            
            # 扩展检索到的状态以匹配序列长度
            retrieved_expanded = retrieved_states.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 融合原始输出和检索知识
            fusion_weight = fusion_weight.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            fused_output = (1 - fusion_weight) * hidden_states + fusion_weight * retrieved_expanded
        else:
            fused_output = hidden_states
        
        return fused_output
    
    def add_to_memory(self, hidden_states: torch.Tensor, labels: torch.Tensor, task_id: int):
        """添加到记忆库"""
        self.amr_core.add_memory(task_id, hidden_states, labels)
    
    def compute_knowledge_distillation_loss(self, student_output: torch.Tensor, 
                                          teacher_output: torch.Tensor) -> torch.Tensor:
        """计算知识蒸馏损失"""
        return self.amr_core.compute_distillation_loss(student_output, teacher_output)