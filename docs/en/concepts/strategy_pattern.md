# Strategy Pattern

The Strategy Pattern is the core architectural pattern that makes BERT-CLAM flexible and modular. This document explains how to add your own continual learning strategies.

## Overview

The Strategy Pattern allows you to define a family of algorithms, encapsulate each one, and make them interchangeable. In BERT-CLAM, each continual learning technique is implemented as a strategy that can be applied to the model's hidden states.

## Adding a Custom Strategy

Adding a custom continual learning strategy takes 5 simple steps:

### Step 1: Implement the Strategy Class

Create a new strategy class that inherits from `ContinualLearningStrategy`:

```python
from bert_clam.core.strategy import ContinualLearningStrategy

class MyCustomStrategy(ContinualLearningStrategy):
    def __init__(self, my_module):
        super().__init__("MyCustom")
        self.module = my_module
    
    def apply(self, hidden_states, model_output, task_id, **kwargs):
        enhanced_states = self.module(hidden_states)
        return enhanced_states, None
```

The `apply` method should:
- Take the current hidden states, model output, and task ID as input
- Apply the strategy's logic to the hidden states
- Return the enhanced hidden states and an optional loss tensor
- Use the `**kwargs` parameter to access additional context (like task memory, embeddings, etc.)

### Step 2: Register in Strategy Factory

You can register your strategy by importing it in your experiment script and using it directly, or by adding it to a strategy factory if you implement one.

### Step 3: Update Model Initialization

When creating your model, add your custom strategy to the strategies list:

```python
from bert_clam.models.bert_clam_model import BERTCLAMModel

# Create your strategy instance
my_strategy = MyCustomStrategy(my_module)

# Create the model without enabling built-in modules
model = BERTCLAMModel(
    model_name='bert-base-uncased',
    num_labels=2,
    lora_r=8,
    lora_alpha=16,
    device=device,
    lora_enabled=True,
    enable_ewc=False,      # Will be controlled by strategy
    enable_amr=False,      # Will be controlled by strategy
    enable_grammar=False,  # Will be controlled by strategy
    strategies=[my_strategy]  # Add your strategy
)
```

### Step 4: Configure in JSON (Optional)

If you want to configure your strategy through JSON files, you'll need to extend the configuration loading mechanism:

```json
{
  "strategies": [
    {"type": "my_custom", "enabled": true}
  ]
}
```

### Step 5: Run Experiment

```bash
python run_experiment.py --config configs/my_custom_experiment.json
```

## Strategy Interface

All strategies must implement the `ContinualLearningStrategy` abstract base class:

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch

class ContinualLearningStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self,
              hidden_states: torch.Tensor,
              model_output: Dict[str, Any],
              task_id: int,
              **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply the strategy to hidden states
        
        Args:
            hidden_states: Current hidden states [batch, seq_len, hidden_size]
            model_output: Model output dictionary (with attentions, pooled_output, etc.)
            task_id: Current task ID
            **kwargs: Additional parameters (task_memory, task_embeddings, etc.)
            
        Returns:
            (enhanced_hidden_states, optional_loss)
        """
        pass
```

## Existing Strategies

### EWC Strategy

The EWC (Elastic Weight Consolidation) strategy prevents catastrophic forgetting by adding a regularization term:

```python
class EWCStrategy(ContinualLearningStrategy):
    def __init__(self, ewc_module, model):
        super().__init__("EWC")
        self.ewc = ewc_module
        self.model = model
    
    def apply(self, hidden_states, model_output, task_id, **kwargs):
        # EWC doesn't modify hidden states, only computes regularization loss
        ewc_loss = None
        if self.ewc and hasattr(self.ewc, 'compute_multi_task_ewc_loss'):
            task_memory = kwargs.get('task_memory', {})
            active_tasks = list(task_memory.keys())
            if active_tasks:
                ewc_loss = self.ewc.compute_multi_task_ewc_loss(self.model, active_tasks)
        
        return hidden_states, ewc_loss
```

### Memory Replay Bank (MRB) Strategy

The MRB strategy retrieves and fuses knowledge from past tasks:

```python
class MRBStrategy(ContinualLearningStrategy):
    def __init__(self, mrb_module, fusion_weight: float = 0.2):
        super().__init__("MRB")
        self.mrb = mrb_module
        self.fusion_weight = fusion_weight
    
    def apply(self, hidden_states, model_output, task_id, **kwargs):
        task_memory = kwargs.get('task_memory', {})
        
        if self.mrb and task_id in task_memory:
            retrieved_knowledge = self.mrb(hidden_states, task_id)
            fused_output = (1 - self.fusion_weight) * hidden_states + self.fusion_weight * retrieved_knowledge
            return fused_output, None
        
        return hidden_states, None
```

### Adaptive LoRA Pooling (ALP) Strategy

The ALP strategy dynamically combines task-specific adaptations:

```python
class ALPStrategy(ContinualLearningStrategy):
    def __init__(self, alp_module):
        super().__init__("ALP")
        self.alp = alp_module
    
    def apply(self, hidden_states, model_output, task_id, **kwargs):
        task_embeddings = kwargs.get('task_embeddings', {})
        get_task_embedding = kwargs.get('get_task_embedding')
        input_ids = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask')
        
        if self.alp and task_id in task_embeddings and get_task_embedding is not None:
            task_embedding = get_task_embedding(input_ids, attention_mask)
            enhanced_output = self.alp(
                hidden_states,
                task_embedding,
                'classifier',
                task_id
            )
            return enhanced_output, None
        
        return hidden_states, None
```

### Grammar Strategy

The Grammar strategy leverages syntactic structure for better generalization:

```python
class GrammarStrategy(ContinualLearningStrategy):
    def __init__(self, grammar_module):
        super().__init__("Grammar")
        self.grammar = grammar_module
    
    def apply(self, hidden_states, model_output, task_id, **kwargs):
        if not self.grammar:
            return hidden_states, None
        
        # Get attention weights
        attentions = model_output.get('attentions')
        attention_weights = attentions[-1] if attentions else None
        
        # Apply grammar-aware enhancement
        if attention_weights is not None:
            enhanced_output = self.grammar(hidden_states, attention_weights)
        else:
            enhanced_output = self.grammar(hidden_states)
        
        # Compute grammar loss
        grammar_loss = None
        if hasattr(self.grammar, 'compute_syntax_aware_loss'):
            grammar_loss = self.grammar.compute_syntax_aware_loss(enhanced_output)
        
        return enhanced_output, grammar_loss
```

## Best Practices

1. **Keep strategies focused**: Each strategy should implement a single, well-defined continual learning technique.

2. **Use the kwargs parameter**: Pass additional context through the `**kwargs` parameter rather than hardcoding dependencies.

3. **Handle optional components gracefully**: Check if modules exist before using them, and return appropriate defaults when they don't.

4. **Minimize state in strategies**: Strategies should be stateless when possible, with state managed by the model or external components.

5. **Consider computational efficiency**: Strategies are applied during each forward pass, so optimize for speed when possible.

The Strategy Pattern makes BERT-CLAM highly flexible, allowing researchers to easily experiment with different combinations of continual learning techniques without modifying the core model code.