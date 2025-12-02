"""
BERT-CLAM模型
完整的基于BERT的持续学习框架
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from bert_clam.models.bert_backbone import EnhancedBERTBackbone
from bert_clam.models.lora_adapter import BERTLoRAModifier
from bert_clam.core.amr import EnhancedAdaptiveMemoryRetrieval
from bert_clam.core.ewc import EnhancedElasticWeightConsolidation
from bert_clam.core.alp import EnhancedAdaptiveLoRAPooling
from bert_clam.core.grammar_aware import EnhancedGrammarAwareModule


class BERTCLAMModel(nn.Module):
    """BERT-CLAM完整模型"""
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_labels: int = 2,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 amr_k: int = 10,
                 ewc_lambda: float = 0.15,
                 alp_top_k: int = 3,
                 grammar_features_dim: int = 64,
                 device: str = None,
                 lora_enabled: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lora_enabled = lora_enabled
        
        # BERT骨干网络
        self.backbone = EnhancedBERTBackbone(
            model_name=model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            use_adapter=False  # 使用LoRA而不是适配器
        )
        
        # 从骨干网络配置中动态获取维度信息
        config = self.backbone.bert.bert.config
        self.hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        
        # LoRA适配器
        if self.lora_enabled:
            self.lora_modifier = BERTLoRAModifier(
                base_model=self.backbone,
                target_modules=['query', 'key', 'value', 'dense', 'classifier'],
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                max_tasks=10
            )
        else:
            self.lora_modifier = None
            # 如果不使用LoRA，确保所有参数都是可训练的
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        # 核心组件
        # 1. 自适应记忆检索 (AMR)
        self.amr = EnhancedAdaptiveMemoryRetrieval(
            hidden_size=self.hidden_size,
            k=amr_k,
            memory_dim=self.hidden_size
        )
        
        # 2. 弹性权重巩固 (EWC)
        self.ewc = EnhancedElasticWeightConsolidation(
            lambda_ewc=ewc_lambda,
            fisher_samples=100,
            update_strategy="cumulative"
        )
        
        # 3. 自适应LoRA池化 (ALP)
        self.alp = EnhancedAdaptiveLoRAPooling(
            hidden_size=self.hidden_size,
            r=lora_r,
            alpha=lora_alpha,
            top_k=alp_top_k,
            similarity_threshold=0.7,
            adaptive_scaling=True,
            task_similarity_aware=True,
            device=self.device
        )
        
        # 4. 语法感知模块
        self.grammar_aware = EnhancedGrammarAwareModule(
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            grammar_features_dim=grammar_features_dim
        )
        
        # 损失权重
        self.loss_weights = {
            'ce': 1.0,      # 交叉熵
            'distill': 0.6, # 知识蒸馏
            'ewc': 0.15,    # EWC正则化
            'grammar': 0.1  # 语法感知
        }
        
        # 任务跟踪
        self.current_task_id = 0
        self.task_embeddings = {}
        self.task_memory = {}
        
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                task_id: int = 0) -> Dict[str, Any]:
       """
       前向传播
       
       Args:
           input_ids: 输入token IDs
           attention_mask: 注意力掩码
           token_type_ids: token类型IDs
           labels: 标签（用于训练）
           task_id: 任务ID
           
       Returns:
           Dict包含预测结果、损失等
       """
       # 如果启用了LoRA，设置当前任务
       if self.lora_enabled and self.lora_modifier:
           self.lora_modifier.set_task(task_id)
       
       # BERT编码
       backbone_outputs = self.backbone(
           input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids,
           output_attentions=True, # 确保输出注意力权重
           output_hidden_states=True # 确保输出隐藏状态
       )
       
       # 获取序列输出和池化输出
       sequence_output = backbone_outputs['sequence_output']
       pooled_output = backbone_outputs['pooled_output']
        
       # 此处不再需要对中间logits进行维度修正，
       # 因为最终的 final_logits 会确保正确的维度。
        
       # 语法感知增强
       if 'attentions' in backbone_outputs and backbone_outputs['attentions']:
           attention_weights = backbone_outputs['attentions'][-1]
           enhanced_output = self.grammar_aware(sequence_output, attention_weights)
       else:
           enhanced_output = self.grammar_aware(sequence_output)
        
       # 应用自适应LoRA池化
       task_embedding = self.get_task_embedding(input_ids, attention_mask)
       if task_id in self.task_embeddings:
           enhanced_output = self.alp(
               enhanced_output,
               self.task_embeddings[task_id],
               'classifier',
               task_id
           )
        
       # AMR知识检索和融合
       if task_id in self.task_memory:
           retrieved_knowledge = self.amr(enhanced_output, task_id)
           fused_output = 0.8 * enhanced_output + 0.2 * retrieved_knowledge
       else:
           fused_output = enhanced_output
        
       # 最终分类 - 使用融合后的输出
       final_pooled = fused_output.mean(dim=1) if fused_output.dim() > 2 else fused_output
       final_logits = self.backbone.bert.classifier(
           self.backbone.bert.pooler_activation(
               self.backbone.bert.pooler(final_pooled)
           )
       )
        
       # 确保logits形状正确 [batch_size, num_labels]
       if final_logits.dim() > 2:
           final_logits = final_logits.mean(dim=1)
        
       if final_logits.size(-1) != self.num_labels:
           raise ValueError(f"Logits维度错误: got {final_logits.size(-1)}, expected {self.num_labels}")
        
       # 构建输出字典
       outputs = {
           'logits': final_logits,
           'sequence_output': fused_output,
           'pooled_output': pooled_output,
           'hidden_states': backbone_outputs.get('hidden_states'),
           'attentions': backbone_outputs.get('attentions')
       }
        
       # 计算损失（如果提供标签）
       if labels is not None:
           # 确保labels是long类型
           labels = labels.long()
            
           # 主要损失
           try:
               ce_loss = nn.functional.cross_entropy(final_logits, labels)
           except RuntimeError as e:
               print(f"Error in cross_entropy: {e}")
               print(f"final_logits shape: {final_logits.shape}, dtype: {final_logits.dtype}")
               print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")
               print(f"labels values: min={labels.min()}, max={labels.max()}")
               raise
           total_loss = self.loss_weights['ce'] * ce_loss
            
           # AMR知识蒸馏损失 - KL散度
           if task_id > 0 and len(self.task_memory) > 0:
               retrieved_knowledge = self.amr(sequence_output, task_id)
               pooled_current = fused_output.mean(dim=1)
               pooled_retrieved = retrieved_knowledge.mean(dim=1)
               distill_loss = self.amr.amr_core.compute_distillation_loss(pooled_current, pooled_retrieved.detach())
               total_loss += self.loss_weights['distill'] * distill_loss
            
           # EWC正则化损失
           if hasattr(self.ewc, 'compute_multi_task_ewc_loss'):
               ewc_loss = self.ewc.compute_multi_task_ewc_loss(
                   self, list(self.task_memory.keys())
               )
               total_loss += self.loss_weights['ewc'] * ewc_loss
            
           # 语法感知损失
           if hasattr(self.grammar_aware, 'compute_syntax_aware_loss'):
               # Ensure fused_output is 3D for grammar_aware module
               if fused_output.dim() > 3:
                   fused_output_3d = fused_output.view(fused_output.size(0), fused_output.size(1), -1)
               elif fused_output.dim() < 3:
                    # Handle cases where the dimension is less than 3, maybe by unsqueezing
                    while fused_output.dim() < 3:
                        fused_output = fused_output.unsqueeze(0)
                    fused_output_3d = fused_output
               else:
                   fused_output_3d = fused_output
               
               # Ensure the last dimension matches hidden_size
               if fused_output_3d.size(2) != self.hidden_size:
                    fused_output_3d = fused_output_3d[:, :, :self.hidden_size]

               grammar_loss = self.grammar_aware.compute_syntax_aware_loss(fused_output_3d)
               total_loss += self.loss_weights['grammar'] * grammar_loss
            
           outputs['loss'] = total_loss
           outputs['ce_loss'] = ce_loss
        
       return outputs
    
    def get_task_embedding(self, 
                          input_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取任务嵌入"""
        with torch.no_grad():
            task_emb = self.backbone.get_task_embedding(input_ids, attention_mask)
            return task_emb
    
    def register_task(self, task_id: int, task_data_loader = None):
        """注册新任务"""
        if task_id not in self.task_embeddings:
            # 为新任务添加LoRA适配器
            if self.lora_enabled and self.lora_modifier:
                self.lora_modifier.add_task(task_id)
            
            # 初始化任务记忆
            self.task_memory[task_id] = {
                'samples': [],
                'embeddings': []
            }
    
    def update_memory(self,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     labels: torch.Tensor,
                     task_id: int):
        """更新记忆库"""
        with torch.no_grad():
            hidden_states = self.backbone.bert.get_hidden_states(input_ids, attention_mask)
            
            # 添加到AMR记忆库
            if self.amr is not None:
                self.amr.add_to_memory(hidden_states, labels, task_id)
            
            # 更新任务嵌入
            task_embedding = self.get_task_embedding(input_ids, attention_mask)
            self.task_embeddings[task_id] = task_embedding
            
            # 添加到ALP任务LoRA
            if self.alp is not None:
                self.alp.add_task_lora(task_id, task_embedding)
    
    def compute_ewc_loss(self, task_ids: list = None) -> torch.Tensor:
        """计算EWC损失"""
        if task_ids is None:
            task_ids = list(self.task_memory.keys())
        
        if hasattr(self.ewc, 'compute_multi_task_ewc_loss'):
            return self.ewc.compute_multi_task_ewc_loss(self, task_ids)
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def save_task_checkpoint(self, task_id: int, dataloader = None):
        """保存任务检查点（用于EWC）"""
        if dataloader is not None and hasattr(self.ewc, 'save_task_data') and self.ewc is not None:
            # 保存当前训练状态
            was_training = self.training
            self.ewc.save_task_data(self, dataloader, task_id)
            # 恢复训练状态
            if was_training:
                self.train()
            else:
                self.eval()
    
    def get_grammar_features(self, input_ids: torch.Tensor, 
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """获取语法特征"""
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask)
            sequence_output = outputs['sequence_output']
            return self.grammar_aware.grammar_core.extract_grammar_features(sequence_output)
