# BERT-CLAM 环境配置指南

本文档提供了经过测试和验证的 `bert-clam` 库的兼容环境配置。

## 推荐配置

- **Python**: 3.11
- **PyTorch**: 2.2.1+cpu
- **TorchVision**: 0.17.1+cpu
- **Transformers**: 4.38.2
- **Tokenizers**: 0.15.2
- **NumPy**: 1.26.4

## 详细依赖列表

完整的兼容依赖列表可以在 `requirements-lock.txt` 文件中找到。

## 安装步骤

为确保环境的完全可复现性，强烈建议使用 `requirements-lock.txt` 文件进行安装。

1. **创建并激活 Conda 环境**
   ```bash
   conda create -n bert_clam python=3.11 -y
   conda activate bert_clam
   ```

2. **使用 `requirements-lock.txt` 安装所有依赖**
   ```bash
   pip install -r requirements-lock.txt
   ```
   *注意：此命令会自动安装所有经过测试的兼容版本的库，包括 PyTorch (CPU 版本)。*

3. **以可编辑模式安装 `bert-clam` 库**
   ```bash
   pip install -e .
   ```

## 故障排除

- **`sentence-transformers` 或 `trl` 冲突**: 不要手动安装 `sentence-transformers` 或 `trl`，它们可能会引入不兼容的 `transformers` 版本。
- **NumPy 兼容性**: `requirements-lock.txt` 已将 NumPy 固定在 `1.26.4` 版本，以避免与 PyTorch 2.2.1 的兼容性问题。
- **`faiss-gpu` vs `faiss-cpu`**: `requirements-lock.txt` 默认安装 `faiss-cpu`。如果您的环境支持 GPU，可以手动安装 `faiss-gpu` 以获得更好的性能。
- **安装失败**: 如果 `pip install` 失败，请确保您的 `pip` 是最新的 (`pip install --upgrade pip`)，并检查网络连接。