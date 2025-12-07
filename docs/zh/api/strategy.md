# Core Strategies API

This section documents the core strategy classes in the BERT-CLAM framework.

## ContinualLearningStrategy

The abstract base class for all continual learning strategies.

### Methods

#### `__init__(name: str)`
Initializes the strategy with a name.

**Parameters:**
- `name` (str): The name of the strategy

#### `apply(hidden_states, model_output, task_id, **kwargs)`
Abstract method to apply the strategy to hidden states.

**Parameters:**
- `hidden_states` (torch.Tensor): Current hidden states [batch, seq_len, hidden_size]
- `model_output` (Dict[str, Any]): Model output dictionary (with attentions, pooled_output, etc.)
- `task_id` (int): Current task ID
- `**kwargs`: Additional parameters (task_memory, task_embeddings, etc.)

**Returns:**
- Tuple[torch.Tensor, Optional[torch.Tensor]]: (enhanced_hidden_states, optional_loss)

## EWCStrategy

The Elastic Weight Consolidation strategy prevents catastrophic forgetting through regularization.

### Methods

#### `__init__(ewc_module, model)`
Initializes the EWC strategy.

**Parameters:**
- `ewc_module`: The EWC module implementation
- `model`: The BERT-CLAM model instance

#### `apply(hidden_states, model_output, task_id, **kwargs)`
Applies the EWC strategy.

The EWC strategy does not modify the hidden states but computes a regularization loss to prevent catastrophic forgetting.

**Returns:**
- Tuple[torch.Tensor, Optional[torch.Tensor]]: (original_hidden_states, ewc_loss)

## MRBStrategy

The Memory Replay Bank strategy retrieves and fuses knowledge from past tasks.

### Methods

#### `__init__(mrb_module, fusion_weight: float = 0.2)`
Initializes the MRB strategy.

**Parameters:**
- `mrb_module`: The MRB module implementation
- `fusion_weight` (float): Weight for fusing retrieved knowledge with current states

#### `apply(hidden_states, model_output, task_id, **kwargs)`
Applies the MRB strategy.

**Returns:**
- Tuple[torch.Tensor, Optional[torch.Tensor]]: (fused_hidden_states, None)

## ALPStrategy

The Adaptive LoRA Pooling strategy dynamically combines task-specific adaptations.

### Methods

#### `__init__(alp_module)`
Initializes the ALP strategy.

**Parameters:**
- `alp_module`: The ALP module implementation

#### `apply(hidden_states, model_output, task_id, **kwargs)`
Applies the ALP strategy.

**Returns:**
- Tuple[torch.Tensor, Optional[torch.Tensor]]: (enhanced_hidden_states, None)

## GrammarStrategy

The Grammar-Aware strategy leverages syntactic structure for better generalization.

### Methods

#### `__init__(grammar_module)`
Initializes the Grammar strategy.

**Parameters:**
- `grammar_module`: The grammar module implementation

#### `apply(hidden_states, model_output, task_id, **kwargs)`
Applies the Grammar strategy.

**Returns:**
- Tuple[torch.Tensor, Optional[torch.Tensor]]: (enhanced_hidden_states, optional_grammar_loss)