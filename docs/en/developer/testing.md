# BERT-CLAM 测试和示例指南

本文档说明如何运行单元测试和使用示例。

## 单元测试

### 运行所有测试

```bash
python -m unittest discover tests/
```

### 运行特定测试文件

```bash
# 测试记忆重放库
python -m unittest tests/test_memory_replay_bank.py

# 测试EWC模块
python -m unittest tests/test_ewc.py

# 测试策略模块
python -m unittest tests/test_strategies.py
```

### 测试覆盖

当前测试覆盖以下核心组件：

- ✅ **记忆重放库 (Memory Replay Bank)**: 测试添加、检索和知识蒸馏功能
- ✅ **弹性权重巩固 (EWC)**: 测试Fisher矩阵计算和EWC惩罚项
- ✅ **策略模式**: 测试EWC、MRB和Grammar策略的应用

## 使用示例

### Jupyter Notebook 快速入门

1. 启动Jupyter Notebook：
```bash
jupyter notebook examples/01_framework_quickstart.ipynb
```

2. 按顺序执行所有单元格，学习如何：
   - 初始化BERT-CLAM模型
   - 创建持续学习策略
   - 训练多个任务
   - 评估模型性能

### Python脚本测试

运行示例测试脚本验证框架功能：

```bash
python examples/test_notebook.py
```

该脚本会：
- 创建虚拟数据
- 初始化模型和策略
- 执行小规模训练
- 验证推理功能

## 测试结果

### 单元测试输出示例

```
..........
----------------------------------------------------------------------
Ran 10 tests in 0.223s

OK
```

### 示例脚本输出

```
=== 测试Notebook代码 ===

1. 初始化tokenizer和设备...
   设备: cpu

2. 创建虚拟数据...
   数据批次: 2

3. 初始化核心模块...
   ✓ 模块初始化完成

4. 创建模型和策略...
   ✓ 策略数量: 3

5. 执行训练测试...
   步骤 1: 损失=1.3397
   步骤 2: 损失=1.2980

6. 执行推理测试...
   预测形状: torch.Size([8])
   预测值: [1, 1, 1, 1]

✓ 所有测试通过！Notebook代码可正常运行。
```

## 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'bert_clam'**
   
   解决方案：安装包
   ```bash
   pip install -e .
   ```

2. **ImportError: faiss is required**
   
   解决方案：安装FAISS
   ```bash
   pip install faiss-cpu  # CPU版本
   # 或
   pip install faiss-gpu  # GPU版本
   ```

3. **CUDA out of memory**
   
   解决方案：减小批次大小或使用CPU
   ```python
   device = torch.device('cpu')
   ```

## 下一步

- 查看 [README.md](README.md) 了解完整文档
- 探索 [configs/](configs/) 目录中的配置示例
- 在真实数据集上测试框架
- 自定义您自己的持续学习策略

## 贡献

如果您发现bug或有改进建议，请：
1. 添加相应的单元测试
2. 确保所有现有测试通过
3. 提交Pull Request

---

**项目状态**: ✅ 所有核心功能已测试并验证