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

### Reproducible Experiments

All experiments are driven by JSON configuration files. Run experiments with a single command:

```bash
python run_experiment.py --config configs/example_strategy_config.json
```
