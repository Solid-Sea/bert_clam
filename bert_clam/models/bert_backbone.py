"""
BERT骨干网络模块
基于Hugging Face Transformers的BERT实现
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Optional, Tuple, Dict, Any


class BERTBackbone(nn.Module):
    """BERT骨干网络"""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 num_labels: int = 2,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1):
        super().__init__()
        
        config = BertConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            output_attentions=True,
            output_hidden_states=True
        )
        config.num_labels = num_labels
        
        self.bert = BertModel.from_pretrained(model_name, config=config, add_pooling_layer=True)
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                task_id: int = 0) -> Dict[str, Any]:
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        logits = self.classifier(pooled_output)
        
        result = {
            'logits': logits,
            'sequence_output': sequence_output,
            'pooled_output': pooled_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
        
        return result

    def get_hidden_states(self, 
                         input_ids: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return outputs.last_hidden_state

class EnhancedBERTBackbone(nn.Module):
    """增强版BERT骨干网络，支持持续学习特性"""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 num_labels: int = 2,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 use_adapter: bool = False,
                 adapter_dim: int = 64):
        super().__init__()
        
        self.bert = BERTBackbone(
            model_name=model_name,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        
        self.use_adapter = use_adapter
        self.adapter_dim = adapter_dim
        
        if use_adapter:
            self.task_adapters = nn.ModuleDict()
            self.task_id_to_adapter = {}
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.bert.hidden_size, self.bert.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.bert.hidden_size // 2, self.bert.hidden_size // 4),
            nn.LayerNorm(self.bert.hidden_size // 4)
        )
    
    def add_task_adapter(self, task_id: int):
        if self.use_adapter:
            adapter = nn.Sequential(
                nn.Linear(self.bert.hidden_size, self.adapter_dim),
                nn.ReLU(),
                nn.Linear(self.adapter_dim, self.bert.hidden_size),
                nn.Dropout(0.1)
            )
            self.task_adapters[str(task_id)] = adapter
            self.task_id_to_adapter[task_id] = str(task_id)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                task_id: int = 0,
                output_attentions: bool = False,
                output_hidden_states: bool = False) -> Dict[str, Any]:
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_id=task_id,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        if self.use_adapter and str(task_id) in self.task_adapters:
            sequence_output = outputs['sequence_output']
            adapter_output = self.task_adapters[str(task_id)](sequence_output)
            outputs['sequence_output'] = sequence_output + adapter_output
            
            pooled_output = outputs['pooled_output']
            pooled_adapter = self.task_adapters[str(task_id)](
                pooled_output.unsqueeze(1)
            ).squeeze(1)
            outputs['pooled_output'] = pooled_output + pooled_adapter
        
        sequence_features = self.feature_extractor(outputs['sequence_output'])
        pooled_features = self.feature_extractor(outputs['pooled_output'].unsqueeze(1)).squeeze(1)
        
        outputs['sequence_features'] = sequence_features
        outputs['pooled_features'] = pooled_features
        
        return outputs
    
    def get_task_embedding(self, 
                          input_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['pooled_output'].mean(dim=0)