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

# 5. 安装其他依赖
pip install -r requirements.txt

# 6. 安装 BERT-CLAM 库
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