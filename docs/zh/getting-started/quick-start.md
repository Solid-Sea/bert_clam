# 快速入门

本指南将帮助您快速开始使用 BERT-CLAM 框架。

## 安装

首先，安装框架：

```bash
conda create -n bert_clam python=3.11 -y
conda activate bert_clam
pip install -r requirements-lock.txt
pip install -e .
```

## 最小示例

以下是一个最小示例，帮助您开始使用：

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

## 运行完整实验

要运行使用预定义配置的完整实验：

```bash
python run_experiment.py --config configs/ablation_full_model.json
```

## 配置驱动的实验

BERT-CLAM 中的所有实验都由 JSON 配置文件驱动。您可以使用单个命令运行任何实验：

```bash
python run_experiment.py --config configs/example_strategy_config.json
```

框架在 `configs/` 目录中提供了几个示例配置：

- `ablation_baseline.json`: 无持续学习
- `ablation_ewc_only.json`: 仅使用 EWC
- `ablation_full_model.json`: 所有策略组合
- `example_strategy_config.json`: 自定义策略排序

## 下一步

现在您已经运行了第一个示例，您可以：

1. 探索[核心概念](../concepts/architecture.md)了解框架架构
2. 尝试[端到端示例](../tutorials/notebook_guide.md)教程
3. 实验不同的持续学习策略
4. 按照[策略模式指南](../concepts/strategy_pattern.md)创建您自己的自定义策略