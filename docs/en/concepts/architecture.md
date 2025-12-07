# Architecture Deep Dive

This section provides a detailed overview of the BERT-CLAM framework architecture.

![BERT-CLAM Core Architecture](../assets/diagrams/architecture.png)

## Strategy Pattern Implementation

The framework uses a clean strategy pattern where each continual learning technique is encapsulated in its own strategy class:

- `ContinualLearningStrategy`: Abstract base class
- `EWCStrategy`, `MRBStrategy`, `ALPStrategy`, `GrammarStrategy`: Concrete implementations
- `BERTCLAMModel`: Orchestrates strategy execution

The strategy pattern allows for flexible combination of different continual learning techniques without modifying core model code.

## Execution Flow

The framework processes data through a chain of strategies:

```python
current_output = sequence_output
for strategy in self.strategies:
    current_output, strategy_loss = strategy.apply(
        hidden_states=current_output,
        model_output=backbone_outputs,
        task_id=task_id
    )
```

This design enables:

1. **Modularity**: Each strategy is self-contained and can be enabled/disabled independently
2. **Composability**: Strategies can be combined in any order
3. **Extensibility**: New strategies can be added without modifying existing code

## Core Components

### BERT Backbone

The framework is built on top of BERT and includes:

- Enhanced BERT backbone with additional features
- Support for various BERT-based models
- Proper handling of hidden states and attention weights

### LoRA Integration

- Parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA)
- Multi-task LoRA support for continual learning scenarios
- Dynamic adapter management for different tasks

### Memory Systems

The framework includes several memory-based components:

- **Memory Replay Bank (MRB)**: Stores and retrieves knowledge from past tasks
- **Task Memory**: Maintains representations of previously learned tasks
- **Adaptive Memory**: Dynamically adjusts memory usage based on task requirements

### Continual Learning Modules

- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting through regularization
- **Adaptive LoRA Pooling (ALP)**: Dynamically combines task-specific adaptations
- **Grammar-Aware Attention**: Leverages syntactic structure for better generalization

## Model Configuration

The `BERTCLAMModel` can be configured with various parameters:

```python
model = BERTCLAMModel(
    model_name='bert-base-uncased',      # Base model
    num_labels=2,                        # Number of output labels
    lora_r=8,                           # LoRA rank
    lora_alpha=16,                      # LoRA scaling factor
    amr_k=10,                           # Memory retrieval size
    ewc_lambda=0.15,                    # EWC regularization strength
    alp_top_k=3,                        # Top-k for ALP
    grammar_features_dim=64,            # Grammar feature dimension
    device=device,                      # Computation device
    lora_enabled=True,                  # Enable LoRA
    enable_ewc=False,                   # Enable EWC (will be controlled by strategy)
    enable_amr=False,                   # Enable AMR (will be controlled by strategy)
    enable_alp=False,                   # Enable ALP (will be controlled by strategy)
    enable_grammar=False,               # Enable grammar (will be controlled by strategy)
    strategies=[]                       # List of strategies to apply
)
```

## Task Management

The framework provides mechanisms for managing multiple tasks:

- `register_task(task_id)`: Register a new task and initialize task-specific components
- `update_memory(...)`: Update memory systems with new task data
- `get_task_embedding(...)`: Obtain embeddings representing task characteristics
- `save_task_checkpoint(...)`: Save task-specific information (e.g., for EWC)

## Loss Computation

The framework combines multiple loss components:

- **Cross-entropy loss**: Primary classification loss
- **Strategy-specific losses**: Losses from individual strategies (EWC, grammar-aware, etc.)
- **Knowledge distillation loss**: Loss for preserving knowledge from previous tasks

This architecture enables researchers to easily experiment with different combinations of continual learning techniques while maintaining a clean, modular codebase.