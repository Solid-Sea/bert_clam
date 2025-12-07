# Main Model API

This section documents the main BERT-CLAM model class.

## BERTCLAMModel

The main BERT-CLAM model that combines BERT with continual learning strategies.

### Methods

#### `__init__(model_name='bert-base-uncased', num_labels=2, lora_r=8, lora_alpha=16, amr_k=10, ewc_lambda=0.15, alp_top_k=3, grammar_features_dim=64, device=None, lora_enabled=True, enable_ewc=False, enable_amr=False, enable_alp=False, enable_grammar=False, strategies=None)`
Initializes the BERT-CLAM model with the specified configuration.

**Parameters:**
- `model_name` (str): Name of the pre-trained BERT model to use (default: 'bert-base-uncased')
- `num_labels` (int): Number of output labels for classification (default: 2)
- `lora_r` (int): Rank parameter for LoRA adaptation (default: 8)
- `lora_alpha` (int): Scaling parameter for LoRA (default: 16)
- `amr_k` (int): Number of items to retrieve from memory replay bank (default: 10)
- `ewc_lambda` (float): Regularization strength for EWC (default: 0.15)
- `alp_top_k` (int): Number of top LoRA adapters to use in Adaptive LoRA Pooling (default: 3)
- `grammar_features_dim` (int): Dimension of grammar features (default: 64)
- `device` (str): Device to run the model on (default: automatically detected)
- `lora_enabled` (bool): Whether to enable LoRA adaptation (default: True)
- `enable_ewc` (bool): Whether to enable EWC (default: False, will be controlled by strategy)
- `enable_amr` (bool): Whether to enable AMR (default: False, will be controlled by strategy)
- `enable_alp` (bool): Whether to enable ALP (default: False, will be controlled by strategy)
- `enable_grammar` (bool): Whether to enable grammar awareness (default: False, will be controlled by strategy)
- `strategies` (List[ContinualLearningStrategy]): List of strategies to apply (default: [])

#### `forward(input_ids, attention_mask=None, token_type_ids=None, labels=None, task_id=0)`
Performs a forward pass through the model.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs
- `attention_mask` (torch.Tensor, optional): Attention mask
- `token_type_ids` (torch.Tensor, optional): Token type IDs
- `labels` (torch.Tensor, optional): Ground truth labels for training
- `task_id` (int): ID of the current task

**Returns:**
- Dict[str, Any]: Dictionary containing logits, sequence output, pooled output, hidden states, attentions, and optionally loss

#### `get_task_embedding(input_ids, attention_mask=None)`
Gets a task embedding for the given inputs.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs
- `attention_mask` (torch.Tensor, optional): Attention mask

**Returns:**
- torch.Tensor: Task embedding

#### `register_task(task_id)`
Registers a new task with the model.

**Parameters:**
- `task_id` (int): ID of the task to register

#### `update_memory(input_ids, attention_mask, labels, task_id)`
Updates the memory systems with new task data.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs
- `attention_mask` (torch.Tensor): Attention mask
- `labels` (torch.Tensor): Ground truth labels
- `task_id` (int): ID of the current task

#### `compute_ewc_loss(task_ids=None)`
Computes the EWC loss for the specified tasks.

**Parameters:**
- `task_ids` (List[int], optional): List of task IDs to compute loss for

**Returns:**
- torch.Tensor: EWC loss value

#### `save_task_checkpoint(task_id, dataloader=None)`
Saves a checkpoint for the specified task (used for EWC).

**Parameters:**
- `task_id` (int): ID of the task to save
- `dataloader` (DataLoader, optional): Dataloader for the task

#### `get_grammar_features(input_ids, attention_mask)`
Gets grammar features for the input.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs
- `attention_mask` (torch.Tensor): Attention mask

**Returns:**
- torch.Tensor: Grammar features