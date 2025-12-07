# BERT-CLAM 快速安装指南

## 前提条件

- 已创建名为 `bert_clam` 的 conda 环境（Python 3.11）
- 已激活该环境

## 安装步骤

### 方法1：使用自动化脚本（推荐）

在 `bert_clam_library` 目录下运行：

```bash
# Windows
setup_and_test.bat

# Linux/Mac
bash setup_and_test.sh
```

脚本会自动完成以下操作：
1. 安装 PyTorch 2.1.0 (CPU版本)
2. 安装 Transformers 4.35.0
3. 安装其他依赖
4. 安装 BERT-CLAM 库
5. 验证安装
6. 运行完整测试

### 方法2：手动安装

```bash
# 1. 激活环境
conda activate bert_clam

# 2. 进入库目录
cd bert_clam_library

# 3. 安装 PyTorch (CPU版本)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 4. 安装 Transformers
pip install transformers==4.35.0

# 5. 安装其他依赖（推荐使用锁定版本）
pip install -r requirements-lock.txt

# 6. 安装 BERT-CLAM 库（会自动安装 faiss-cpu 等核心依赖）
pip install -e .
```

## 验证安装

安装完成后，运行以下命令验证：

```python
# 测试基本导入
python -c "import bert_clam; print('BERT-CLAM 版本:', bert_clam.__version__)"
python -c "from bert_clam.training import BERTCLAMTrainer; print('✓ BERTCLAMTrainer 导入成功')"

# 运行完整测试
python complete_test.py
```

## 预期结果

完整测试应该显示：

```
✓ bert_clam 包导入: PASS
✓ BERTCLAMTrainer 导入: PASS
✓ config_loader 模块导入: PASS
✓ lora_adapter 模块导入: PASS
✓ BERT-CLA模型: PASS
✓ COLA配置: PASS
✓ 数据处理: PASS
✓ 模型前向传播: PASS
✓ 训练步骤: PASS
✓ 评估指标: PASS

测试结果: 10/10 个测试通过
成功率: 100.0%
```

## 故障排除

### 问题1：PyTorch 安装失败

```bash
# 尝试使用 conda 安装
conda install pytorch==2.1.0 torchvision==0.16.0 cpuonly -c pytorch
```

### 问题2：Transformers 导入错误

```bash
# 重新安装 transformers
pip uninstall transformers -y
pip install transformers==4.35.0
```

### 问题3：BERT-CLAM 导入失败

```bash
# 重新安装库
pip uninstall bert-clam -y
pip install -e .
```

### 问题4：Hugging Face 模型下载缓慢或失败

如果您在中国大陆地区遇到 Hugging Face 模型下载问题，可以配置镜像源：

```bash
# 临时设置（仅当前会话有效）
export HF_ENDPOINT=https://hf-mirror.com

# 或在 Windows PowerShell 中
$env:HF_ENDPOINT="https://hf-mirror.com"

# 永久设置（Linux/Mac）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 永久设置（Windows）
setx HF_ENDPOINT "https://hf-mirror.com"
```

配置后，所有 Hugging Face 模型和数据集的下载都会通过镜像加速。

### 问题5：依赖版本冲突

如果遇到依赖冲突（特别是 `transformers` 和 `sentence-transformers` 之间），请使用锁定版本文件：

```bash
pip install -r requirements-lock.txt
```

**注意**：`faiss-cpu` 会作为核心依赖自动安装，无需手动安装。

## 环境信息

推荐的环境配置：

```
Python: 3.11
PyTorch: 2.1.0
torchvision: 0.16.0
transformers: 4.35.0
numpy: 1.24.3
pandas: 2.0.3
scikit-learn: 1.3.0
```

## 下一步

安装成功后，您可以：

1. 查看 [README.md](../../README.md) 了解使用方法
2. 运行示例代码测试功能
3. 开始您的持续学习实验

## 获取帮助

如果遇到问题，请查看：
- [ENVIRONMENT_COMPATIBILITY_REPORT.md](../../ENVIRONMENT_COMPATIBILITY_REPORT.md) - 环境兼容性详细信息
- [README.md](../../README.md) - 完整文档