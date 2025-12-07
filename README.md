# BERT-CLAM: A Modular Framework for Continual Learning Research

**BERT-CLAM** is a flexible, production-ready framework for prototyping and comparing continual learning strategies on NLP tasks. Built on BERT and designed with modularity in mind, it enables researchers and developers to quickly experiment with different combinations of continual learning techniques.

## Overview

BERT-CLAM transforms continual learning research from hardcoded experiments into a composable, strategy-driven framework. Instead of modifying model code for each experiment, you simply configure which strategies to apply and in what order.

**Key Value Proposition:**
- Rapid Prototyping: Test new continual learning ideas in hours, not days
- Fair Comparisons: Standardized evaluation across different strategies
- Production-Ready: Clean architecture suitable for real-world deployment

## Core Features

### Modular by Design

BERT-CLAM uses the Strategy Pattern to decouple continual learning techniques from the core model.

### Built-in Strategies

- **EWC**: Elastic Weight Consolidation - Prevents catastrophic forgetting via regularization
- **MRB**: Memory Replay Bank - Retrieves and fuses knowledge from past tasks
- **ALP**: Adaptive LoRA Pooling - Dynamically combines task-specific adaptations
- **Grammar**: Grammar-Aware Attention - Leverages syntactic structure for better generalization

### Why BERT-CLAM?

| 特性 | 传统微调 | EWC | BERT-CLAM |
|------|---------|-----|-----------|
| **灾难性遗忘** 🧠 | ❌ 严重 | ⚠️ 中等 | ✅ 轻微 |
| **新任务适应** 🚀 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **旧任务保持** 🔒 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **计算开销** 💻 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **内存占用** 💾 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **实现复杂度** 🛠️ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **可扩展性** 📈 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **语法一致性** 📝 | ❌ 无保证 | ❌ 无保证 | ✅ 强保证 |

**核心优势**: BERT-CLAM 通过组合多种持续学习策略，在保持旧任务性能的同时，实现了对新任务的快速适应，并通过语法约束确保了输出的一致性和可靠性。

### Reproducible Experiments

All experiments are driven by JSON configuration files. Run experiments with a single command:
```bash
python run_experiment.py --config configs/example_strategy_config.json
```

## Quick Start

### Installation

```bash
conda create -n bert_clam python=3.11 -y
conda activate bert_clam
pip install -r requirements-lock.txt
pip install -e .
```

### Minimal Example

```python
import torch
from transformers import AutoTokenizer
from bert_clam.models.bert_clam_model import BERTCLAMModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BERTCLAMModel(
    model_name='bert-base-uncased',
    num_labels=2,
    enable_grammar=True,
    enable_ewc=True
)

inputs = tokenizer("Example sentence", return_tensors="pt")
outputs = model(**inputs, task_id=0)
print(f"Logits: {outputs['logits']}")
```

### Run a Full Experiment

```bash
python run_experiment.py --config configs/ablation_full_model.json
```

## Architecture Deep Dive

### Strategy Pattern Implementation

The framework uses a clean strategy pattern where each continual learning technique is encapsulated in its own strategy class:

- [`ContinualLearningStrategy`](bert_clam/core/strategy.py:11): Abstract base class
- [`EWCStrategy`](bert_clam/core/strategy.py:28), [`MRBStrategy`](bert_clam/core/strategy.py:45), [`ALPStrategy`](bert_clam/core/strategy.py:68), [`GrammarStrategy`](bert_clam/core/strategy.py:91): Concrete implementations
- [`BERTCLAMModel`](bert_clam/models/bert_clam_model.py:17): Orchestrates strategy execution

### Execution Flow

```python
current_output = sequence_output
for strategy in self.strategies:
    current_output, strategy_loss = strategy.apply(
        hidden_states=current_output,
        model_output=backbone_outputs,
        task_id=task_id
    )
```

## How to Add Your Own Strategy

Adding a custom continual learning strategy takes 5 simple steps:

### Step 1: Implement the Strategy Class

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

### Step 2: Register in Strategy Factory

Edit [`run_experiment.py`](run_experiment.py:31) to add your strategy type.

### Step 3: Update Model Initialization

Add your module to [`BERTCLAMModel`](bert_clam/models/bert_clam_model.py:20).

### Step 4: Configure in JSON

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

## Example Configurations

See [`configs/`](configs/) for examples:
- `ablation_baseline.json`: No continual learning
- `ablation_ewc_only.json`: EWC only
- `ablation_full_model.json`: All strategies
- `example_strategy_config.json`: Custom strategy ordering

## Limitations

Current limitations:
1. Tested primarily on GLUE benchmark subsets
2. Full model requires approximately 2x training time vs baseline
3. MRB strategy increases memory usage
4. Hyperparameters require tuning per dataset

## License

MIT License - see LICENSE file for details.

## 📚 Documentation

To build and serve the documentation locally, run the following command:

```bash
mkdocs serve
```

Then, open your browser to `http://127.0.0.1:8000`.