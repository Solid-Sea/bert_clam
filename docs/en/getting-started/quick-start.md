# Quick Start

This guide will help you get started with the BERT-CLAM framework quickly.

## Installation

First, install the framework:

```bash
conda create -n bert_clam python=3.11 -y
conda activate bert_clam
pip install -r requirements-lock.txt
pip install -e .
```

## Minimal Example

Here's a minimal example to get you started:

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

## Run a Full Experiment

To run a complete experiment with predefined configurations:

```bash
python run_experiment.py --config configs/ablation_full_model.json
```

## Configuration-Driven Experiments

All experiments in BERT-CLAM are driven by JSON configuration files. You can run any experiment with a single command:

```bash
python run_experiment.py --config configs/example_strategy_config.json
```

The framework comes with several example configurations in the `configs/` directory:

- `ablation_baseline.json`: No continual learning
- `ablation_ewc_only.json`: EWC only
- `ablation_full_model.json`: All strategies combined
- `example_strategy_config.json`: Custom strategy ordering

## Next Steps

Now that you've run your first example, you can:

1. Explore the [Core Concepts](../concepts/architecture.md) to understand the framework architecture
2. Try the [End-to-End Example](../tutorials/notebook_guide.md) tutorial
3. Experiment with different continual learning strategies
4. Create your own custom strategies following the [Strategy Pattern guide](../concepts/strategy_pattern.md)