# BERT-CLAM: BERT-based Continual Learning with Adaptive Memory

BERT-CLAM is a continual learning framework built on top of the powerful BERT model. It is designed to learn from a sequence of tasks without catastrophically forgetting previously learned knowledge.

## Key Features

*   **Continual Learning**: Learn from a sequence of tasks without significant performance degradation on previous tasks.
*   **Adaptive Memory**: A sophisticated memory management system that stores and retrieves important samples from previous tasks.
*   **Elastic Weight Consolidation (EWC)**: A regularization-based approach to prevent catastrophic forgetting.
*   **Adaptive LoRA Pooling (ALP)**: A novel technique for pooling LoRA weights from similar tasks to improve performance on new tasks.
*   **Grammar-Aware Attention**: A syntax-aware attention mechanism that enhances the model's understanding of grammatical structures.

## Installation

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n bert_clam python=3.11 -y
    conda activate bert_clam
    ```

2.  **Install all dependencies from the lock file:**
    ```bash
    pip install -r requirements-lock.txt
    ```

3.  **Install the library in editable mode:**
    ```bash
    pip install -e .
    ```
For more detailed instructions and troubleshooting, see `ENVIRONMENT_SETUP.md`.

## Quick Start: A Minimal Example

Here is a minimal example of how to use BERT-CLAM for a simple classification task.

```python
import torch
from transformers import AutoTokenizer
from bert_clam.models.bert_clam_model import BERTCLAMModel

# 1. Configuration
model_name = 'prajjwal1/bert-tiny'
num_labels = 2

# 2. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BERTCLAMModel(model_name=model_name, num_labels=num_labels, lora_enabled=False)
model.eval() # Set to evaluation mode

# 3. Prepare Input Data
sentences = [
    "This is a grammatically correct sentence.",
    "This sentence is not correct grammar."
]
labels = torch.tensor([1, 0])

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 4. Forward Pass
with torch.no_grad():
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        labels=labels
    )

# 5. Print Results
predictions = torch.argmax(outputs['logits'], dim=-1)
accuracy = (predictions == labels).float().mean()

print(f"Logits: {outputs['logits']}")
print(f"Predictions: {predictions}")
print(f"Labels: {labels}")
print(f"Accuracy: {accuracy.item():.2f}")
print(f"Loss: {outputs['loss'].item():.4f}")
```

## Testing

The library includes a comprehensive test suite. To run the tests, execute the following command from the `bert_clam_library` directory:

```bash
python complete_test.py
```

**Current Test Status:** 🎉 **100% (10/10) tests passing!**