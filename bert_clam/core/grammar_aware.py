"""
Grammar-Aware: 语法感知模块
利用BERT注意力机制进行语法感知学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class GrammarAwareModule(nn.Module):
    """语法感知模块 - 利用BERT注意力权重进行语法感知学习"""
    
    def __init__(self, hidden_size: int = 768, num_attention_heads: int = 12,
                 grammar_features_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.grammar_features_dim = grammar_features_dim
        
        # 检查grammar_features_dim是否有效
        if grammar_features_dim <= 0:
            # 如果维度无效，设置为禁用模式
            self.disabled = True
            return
        else:
            self.disabled = False
        
        # 确保LSTM的参数有效
        lstm_hidden_size = max(1, grammar_features_dim // 2)
        
        # 语法特征提取器
        self.grammar_feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_size // 2), grammar_features_dim),
            nn.LayerNorm(grammar_features_dim)
        )
        
        # 注意力权重分析器
        self.attention_analyzer = nn.Sequential(
            nn.Linear(num_attention_heads, max(1, num_attention_heads // 2)),
            nn.ReLU(),
            nn.Linear(max(1, num_attention_heads // 2), max(1, num_attention_heads // 4)),
            nn.Sigmoid()
        )
        
        # 语法感知正则化器
        self.grammar_regularizer = nn.Sequential(
            nn.Linear(grammar_features_dim, max(1, grammar_features_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(1, grammar_features_dim // 2), 1),
            nn.Sigmoid()
        )
        
        # 语法结构识别器 - 确保LSTM参数有效
        self.structure_identifier = nn.LSTM(
            input_size=grammar_features_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 语法感知门控
        self.grammar_gate = nn.Sequential(
            nn.Linear(grammar_features_dim, max(1, grammar_features_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(1, grammar_features_dim // 4), 1),
            nn.Sigmoid()
        )

        # 预定义投影层
        self._grammar_projection = nn.Linear(lstm_hidden_size * 2, hidden_size)
    
    def extract_grammar_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """从隐藏状态中提取语法特征
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            grammar_features: [batch_size, seq_len, grammar_features_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 提取语法特征
        grammar_features = self.grammar_feature_extractor(hidden_states)
        
        return grammar_features
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """分析注意力模式中的语法信息
        
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            grammar_attention: [batch_size, seq_len, seq_len]
        """
        # 将num_heads维度移到最后
        attention_weights = attention_weights.permute(0, 2, 3, 1)  # [batch_size, seq_len, seq_len, num_heads]
        
        # 分析注意力模式
        batch_size, seq_len, _, _ = attention_weights.shape
        
        # 使用attention_analyzer处理每个位置的注意力权重
        # 形状变为 [batch_size, seq_len, seq_len, new_dim]
        attention_features = self.attention_analyzer(attention_weights)
        
        # 简化为语法感知注意力
        # 形状变为 [batch_size, seq_len, seq_len]
        grammar_attention = attention_features.mean(dim=-1)
        
        return grammar_attention
    
    def identify_grammar_structure(self, grammar_features: torch.Tensor) -> torch.Tensor:
        """识别语法结构
        
        Args:
            grammar_features: [batch_size, seq_len, grammar_features_dim]
            
        Returns:
            structure_features: [batch_size, seq_len, grammar_features_dim]
        """
        # 使用LSTM识别序列中的语法结构
        structure_out, _ = self.structure_identifier(grammar_features)
        
        return structure_out
    
    def compute_grammar_regularization(self, grammar_features: torch.Tensor) -> torch.Tensor:
        """计算语法正则化损失
        
        Args:
            grammar_features: [batch_size, seq_len, grammar_features_dim]
            
        Returns:
            regularization_loss: 标量张量
        """
        # 计算语法特征的正则化
        batch_size, seq_len, _ = grammar_features.shape
        
        # 语法一致性正则化
        grammar_reg = self.grammar_regularizer(grammar_features)  # [batch_size, seq_len, 1]
        grammar_reg = grammar_reg.mean()  # 平均正则化损失
        
        return grammar_reg
    
    def forward(self, hidden_states: torch.Tensor,
               attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_weights: [batch_size, num_heads, seq_len, seq_len] or None
            
        Returns:
            enhanced_states: [batch_size, seq_len, hidden_size]
        """
        # 如果模块被禁用，直接返回输入
        if getattr(self, 'disabled', False):
            return hidden_states
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 提取语法特征
        grammar_features = self.extract_grammar_features(hidden_states)  # [batch, seq, grammar_dim]
        
        # 分析注意力模式（如果提供）
        if attention_weights is not None:
            grammar_attention = self.analyze_attention_patterns(attention_weights)
        else:
            # 创建默认的语法感知权重
            grammar_attention = torch.ones(batch_size, seq_len, seq_len, device=hidden_states.device)
        
        # 识别语法结构
        structure_features = self.identify_grammar_structure(grammar_features)
        
        # 计算语法门控
        grammar_gate = self.grammar_gate(grammar_features)  # [batch, seq, 1]
        
        # 融合语法信息
        # 扩展grammar_gate以匹配hidden_states维度
        grammar_gate_expanded = grammar_gate.expand(-1, -1, hidden_size)
        
        # 简化的语法增强：使用预定义的线性层投影语法特征
        self._grammar_projection.to(hidden_states.device)
        expanded_grammar = self._grammar_projection(structure_features)
        
        # 简化的语法增强
        enhanced_states = hidden_states + 0.1 * expanded_grammar
        
        # 应用语法门控
        result = grammar_gate_expanded * enhanced_states + (1 - grammar_gate_expanded) * hidden_states
        
        return result
    
    def get_grammar_regularization_loss(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """获取语法正则化损失"""
        grammar_features = self.extract_grammar_features(hidden_states)
        return self.compute_grammar_regularization(grammar_features)


class EnhancedGrammarAwareModule(nn.Module):
    """增强语法感知模块"""
    
    def __init__(self, hidden_size: int = 768, num_attention_heads: int = 12,
                 grammar_features_dim: int = 64, max_seq_length: int = 512):
        super().__init__()
        
        # 确保grammar_features_dim大于0
        if grammar_features_dim <= 0:
            # 如果grammar_features_dim为0或负数，使用默认值
            grammar_features_dim = 64
        
        self.grammar_core = GrammarAwareModule(hidden_size, num_attention_heads, grammar_features_dim)
        
        # 语法依赖分析器
        self.dependency_analyzer = nn.Sequential(
            nn.Linear(min(max_seq_length, 128), min(max_seq_length, 128) // 2),  # 限制最大尺寸
            nn.ReLU(),
            nn.Linear(min(max_seq_length, 128) // 2, min(max_seq_length, 128) // 4),
            nn.Softmax(dim=-1)
        )
        
        # 语法层次感知器
        self.hierarchy_perceptor = nn.MultiheadAttention(
            embed_dim=grammar_features_dim,
            num_heads=min(4, max(1, grammar_features_dim // 16)),  # 根据embed_dim调整头数
            batch_first=True
        )
        
        # 语法稳定性评估器
        self.stability_evaluator = nn.Sequential(
            nn.Linear(grammar_features_dim, max(32, grammar_features_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(32, grammar_features_dim // 2), 1),
            nn.Sigmoid()
        )

        # 预先定义投影层
        self._hierarchy_projection = nn.Linear(grammar_features_dim, hidden_size)
    
    def forward(self, hidden_states: torch.Tensor,
               attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """增强的前向传播"""
        # 检查基础模块是否被禁用
        if getattr(self.grammar_core, 'disabled', False):
            # 如果基础模块被禁用，直接返回输入
            return hidden_states
        
        # 1. 提取基础语法特征
        grammar_features = self.grammar_core.extract_grammar_features(hidden_states)
        
        # 2. 层次语法感知
        hierarchy_enhanced, _ = self.hierarchy_perceptor(
            grammar_features, grammar_features, grammar_features
        )
        
        # 3. 将层次语法增强映射到hidden_states维度
        self._hierarchy_projection.to(hidden_states.device)
        projected_hierarchy = self._hierarchy_projection(hierarchy_enhanced)
        
        # 4. 基础语法感知处理
        base_enhanced = self.grammar_core(hidden_states, attention_weights)
        
        # 5. 融合所有信息
        final_enhanced = base_enhanced + 0.1 * projected_hierarchy
        
        return final_enhanced
    
    def compute_syntax_aware_loss(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算语法感知损失"""
        return self.grammar_core.get_grammar_regularization_loss(hidden_states)
    
    def analyze_syntax_dependencies(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        """分析语法依赖关系"""
        seq_len = sequence_ids.size(-1)
        if seq_len <= self.dependency_analyzer[0].in_features:
            # 扩展序列以匹配依赖分析器输入维度
            extended_seq = F.pad(sequence_ids.float(), (0, self.dependency_analyzer[0].in_features - seq_len))
            dependencies = self.dependency_analyzer(extended_seq.unsqueeze(0))
            return dependencies.squeeze(0)
        else:
            # 截断序列
            truncated_seq = sequence_ids[:, :self.dependency_analyzer[0].in_features].float()
            return self.dependency_analyzer(truncated_seq)
    
    def evaluate_syntax_stability(self, grammar_features: torch.Tensor) -> torch.Tensor:
        """评估语法稳定性"""
        stability = self.stability_evaluator(grammar_features)
        return stability.mean(dim=1)  # [batch_size]