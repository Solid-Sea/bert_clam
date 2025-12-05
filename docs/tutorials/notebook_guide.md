# End-to-End Example

This tutorial demonstrates how to use the BERT-CLAM framework for continual learning tasks using our example notebook.

## Overview

The example notebook `01_framework_quickstart.ipynb` in the `examples/` directory provides a complete walkthrough of using BERT-CLAM for a continual learning scenario. The notebook covers:

1. Environment setup
2. Data loading
3. Model and module initialization
4. Strategy creation and composition
5. Training with injected strategies
6. Evaluation and results summary

## Running the Example

### Prerequisites

Make sure you have installed the framework following the [Installation guide](../getting-started/installation.md).

### Launch the Notebook

You can run the example notebook in one of these ways:

**Option 1: Jupyter Notebook**
```bash
jupyter notebook examples/01_framework_quickstart.ipynb
```

**Option 2: Jupyter Lab**
```bash
jupyter lab examples/01_framework_quickstart.ipynb
```

## Key Concepts Demonstrated

### 1. Component Initialization

The notebook demonstrates how to initialize the core continual learning modules:

```python
from bert_clam.core.ewc import EnhancedElasticWeightConsolidation
from bert_clam.core.memory_replay_bank import EnhancedAdaptiveMemoryRetrieval
from bert_clam.core.grammar_aware import EnhancedGrammarAwareModule

ewc_module = EnhancedElasticWeightConsolidation(
    lambda_ewc=0.5,
    fisher_samples=10
)

mrb_module = EnhancedAdaptiveMemoryRetrieval(
    hidden_size=768,
    k=5,
    memory_dim=768
)

grammar_module = EnhancedGrammarAwareModule(
    hidden_size=768,
    num_attention_heads=12,
    grammar_features_dim=64
)
```

### 2. Strategy Composition

The notebook shows how to create and compose strategies:

```python
from bert_clam.core.strategy import EWCStrategy, MRBStrategy, GrammarStrategy

strategies = [
    GrammarStrategy(grammar_module.to(device)),
    MRBStrategy(mrb_module.to(device), fusion_weight=0.2),
    EWCStrategy(ewc_module, model)
]

# Inject strategies into the model
model.strategies = strategies
```

### 3. Multi-Task Training

The notebook demonstrates training on multiple tasks sequentially:

```python
def train_task(model, dataloader, task_id, num_steps=5):
    """Train a single task"""
    model.train()
    model.register_task(task_id)
    
    # ... training loop implementation
```

## Understanding the Code Structure

The notebook is organized into these sections:

1. **Environment Setup**: Installing dependencies and importing modules
2. **Data Loading**: Creating dummy data for demonstration (replace with your own dataset)
3. **Model Initialization**: Setting up the BERT-CLAM model and core modules
4. **Strategy Composition**: Creating and combining different continual learning strategies
5. **Training**: Implementing the training loop for multiple tasks
6. **Evaluation**: Assessing model performance across tasks
7. **Summary**: Key takeaways and next steps

## Adapting for Your Use Case

To use the notebook with your own data:

1. Replace the `create_dummy_data()` function with your data loading logic
2. Adjust the `num_labels` parameter based on your classification task
3. Modify the training parameters (learning rate, epochs, etc.) as needed
4. Update the evaluation metrics based on your specific requirements

## Key Takeaways

- BERT-CLAM allows flexible combination of continual learning strategies
- The strategy pattern enables modular design and easy experimentation
- Multiple tasks can be trained sequentially with minimal code changes
- The framework handles task memory and strategy application automatically

## Next Steps

After completing the tutorial, you can:

1. Experiment with different combinations of strategies
2. Apply the framework to your own datasets
3. Create custom strategies following the pattern shown in the [Strategy Pattern guide](../concepts/strategy_pattern.md)
4. Explore the configuration files in the `configs/` directory for more advanced usage