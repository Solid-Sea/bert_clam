"""
BERT-CLAM训练器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
import os
import json
from tqdm import tqdm
import numpy as np
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bert_clam_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 修正导入路径
from bert_clam.models.bert_clam_model import BERTCLAMModel
from bert_clam.data.continual_dataset import ContinualDataset
from bert_clam.evaluation.forgetting_evaluator import ForgettingEvaluator  # 修正导入


class BERTCLAMTrainer:
    """BERT-CLAM训练器"""
    
    def __init__(self, 
                 model: BERTCLAMModel,
                 config: Dict[str, Any],
                 output_dir: str = "./experiments"):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
        # 优化器
        # 优化器 - 只优化可训练的参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"优化器找到了 {len(trainable_params)} 个可训练的参数张量。")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度器
        from transformers import get_linear_schedule_with_warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 100),
            num_training_steps=config.get('num_training_steps', 1000)
        )
        
        # 初始化遗忘评估器
        self.forgetting_evaluator = ForgettingEvaluator()
        
        # 训练历史
        self.training_history = {
            'task_losses': {},
            'task_accuracies': {},
            'forgetting_rates': {},
            'backward_transfers': {}
        }
        
        # 当前任务信息
        self.current_task_id = 0
        self.task_performance_history = {}
        self.task_names = []  # 存储任务名称

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.model.device.type == 'cuda')
    
    def train_task(self,
                   task_id: int,
                   train_loader: DataLoader,
                   val_loader: DataLoader = None,
                   epochs: int = 3,
                   save_checkpoint: bool = True):
        """训练单个任务"""
        logger.info(f"=" * 60)
        logger.info(f"开始训练任务 {task_id}")
        logger.info(f"训练轮数: {epochs}, 批次大小: {train_loader.batch_size}")
        logger.info(f"=" * 60)
        
        try:
            self.model.train()
            
            # 注册新任务
            logger.info(f"注册任务 {task_id}...")
            self.model.register_task(task_id)
            logger.info(f"任务 {task_id} 注册成功")
        except Exception as e:
            logger.error(f"任务注册失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        for epoch in range(epochs):
            self.model.train()  # 确保每个epoch开始时模型处于训练模式
            total_loss = 0
            num_batches = 0
            
            logger.info(f"Epoch {epoch+1}/{epochs} 开始")
            progress_bar = tqdm(train_loader, desc=f"Task {task_id}, Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    input_ids = batch['input_ids'].to(self.model.device)
                    attention_mask = batch['attention_mask'].to(self.model.device)
                    labels = batch['labels'].to(self.model.device)
                    
                    # 确保标签维度正确
                    if labels.dim() == 2 and labels.size(-1) == self.model.num_labels:
                        labels = torch.argmax(labels, dim=-1)
                    elif labels.dim() > 1:
                        labels = labels.squeeze(-1)
                    
                    labels = labels.long()
                    labels = torch.clamp(labels, 0, self.model.num_labels - 1)
                    
                    # 前向传播
                    device_type = 'cuda' if self.model.device.type == 'cuda' else 'cpu'
                    with torch.amp.autocast(device_type, enabled=self.model.device.type == 'cuda'):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            task_id=task_id
                        )
                        
                        if 'loss' in outputs:
                            loss = outputs['loss']
                        else:
                            logits = outputs['logits']
                            loss = nn.functional.cross_entropy(logits, labels)
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.config.get('gradient_clip', 1.0))
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # 更新记忆库
                    with torch.no_grad():
                        self.model.update_memory(
                            input_ids, attention_mask, labels, task_id
                        )
                    
                    # 每100个批次记录一次详细信息
                    if batch_idx % 100 == 0:
                        logger.info(f"Batch {batch_idx}: loss={loss.item():.4f}")
                        
                except Exception as e:
                    logger.error(f"批次 {batch_idx} 训练失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            
            avg_loss = total_loss / num_batches
            logger.info(f"Task {task_id}, Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
            # 验证（如果提供验证数据）
            # 注意: 这里的验证只是训练过程中的监控,不记录到性能历史
            if val_loader:
                try:
                    val_acc = self.evaluate(val_loader, task_id, record_performance=False)
                    logger.info(f"Task {task_id}, Validation Accuracy: {val_acc:.4f}")
                except Exception as e:
                    logger.error(f"验证失败: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # 保存任务检查点（用于EWC）
        try:
            if hasattr(self.model, 'save_task_checkpoint'):
                logger.info(f"保存任务 {task_id} 的EWC检查点...")
                self.model.save_task_checkpoint(task_id, train_loader)
        except Exception as e:
            logger.warning(f"EWC检查点保存失败: {str(e)}")
        
        # 记录任务损失
        self.training_history['task_losses'].setdefault(task_id, []).append(avg_loss)
        
        # 保存检查点
        if save_checkpoint:
            try:
                self.save_checkpoint(task_id)
                logger.info(f"任务 {task_id} 训练完成并保存检查点")
            except Exception as e:
                logger.error(f"检查点保存失败: {str(e)}")
                logger.error(traceback.format_exc())
    
    def evaluate(self,
                data_loader: DataLoader,
                task_id: int = 0,
                compute_forgetting: bool = False,
                record_performance: bool = True) -> float:
        """评估模型
        
        Args:
            data_loader: 数据加载器
            task_id: 任务ID
            compute_forgetting: 是否计算遗忘率(已废弃,保留用于兼容性)
            record_performance: 是否将结果记录到性能历史中
                               True: 记录(用于训练后的验证和遗忘率跟踪)
                               False: 不记录(用于临时评估)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=task_id
                )
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        
        # 只在需要时记录任务性能
        if record_performance:
            if task_id not in self.task_performance_history:
                self.task_performance_history[task_id] = []
            self.task_performance_history[task_id].append(accuracy)
            logger.debug(f"记录Task {task_id}性能: {accuracy:.4f}, 当前历史: {self.task_performance_history[task_id]}")
        
        return accuracy
    
    def evaluate_all_tasks(self, 
                          task_loaders: Dict[int, DataLoader],
                          current_task_id: int) -> Dict[int, float]:
        """评估所有已学习的任务（用于计算遗忘率）"""
        task_accuracies = {}
        
        for task_id, loader in task_loaders.items():
            if task_id <= current_task_id: # 只评估已学习的任务
                acc = self.evaluate(loader, task_id)
                task_accuracies[task_id] = acc
                print(f"Task {task_id} Accuracy: {acc:.4f}")
        
        return task_accuracies
    
    def compute_forgetting_rate(self, task_id: int) -> float:
        """计算特定任务的遗忘率"""
        if task_id not in self.task_performance_history:
            return 0.0
        
        performance_history = self.task_performance_history[task_id]
        if len(performance_history) < 2:
            return 0.0
        
        # 计算最高准确率（除了最终）和最终准确率
        max_acc_before_final = max(performance_history[:-1])
        final_acc = performance_history[-1]
        
        forgetting_rate = max(0.0, max_acc_before_final - final_acc)
        return forgetting_rate
    
    def compute_backward_transfer(self, task_id: int) -> float:
        """计算向后迁移"""
        if task_id <= 0 or task_id not in self.task_performance_history:
            return 0.0
        
        # 当前任务在学习后的性能
        current_task_final_acc = self.task_performance_history[task_id][-1] if self.task_performance_history[task_id] else 0.0
        
        # 前一个任务的性能变化
        prev_task_id = task_id - 1
        if prev_task_id in self.task_performance_history:
            prev_task_initial_acc = self.task_performance_history[prev_task_id][0] if len(self.task_performance_history[prev_task_id]) > 0 else 0.0
            prev_task_final_acc = self.task_performance_history[prev_task_id][-1] if self.task_performance_history[prev_task_id] else 0.0
            
            bwt = prev_task_final_acc - prev_task_initial_acc
            return bwt
        
        return 0.0
    
    def continual_learning_train(self,
                                continual_dataset: ContinualDataset,
                                epochs_per_task: int = 3):
        """持续学习训练流程"""
        print("开始持续学习训练...")
        
        task_loaders = {}
        
        # 获取任务名称
        num_tasks = continual_dataset.get_num_tasks()
        for task_idx in range(num_tasks):
            task_info = continual_dataset.get_task_info(task_idx)  # 假设有这个方法获取任务信息
            if task_info:
                task_name = task_info.get('name', f'task_{task_idx}')
                if task_name not in self.task_names:
                    self.task_names.append(task_name)
        
        # 训练每个任务
        for task_idx in range(num_tasks):
            task_info = continual_dataset.get_next_task(self.config.get('batch_size', 32))
            if task_info is None:
                break
            
            task_name, train_loader, val_loader = task_info
            task_id = task_idx
            
            print(f"\n=== 训练任务 {task_id}: {task_name} ===")
            
            # 训练当前任务
            self.train_task(
                task_id=task_id,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_task
            )
            
            # 评估所有已学习的任务（用于计算遗忘率）
            print(f"\n评估所有已学习的任务以计算遗忘率...")
            for eval_task_idx in range(task_id + 1):  # 评估从0到当前任务的所有任务
                eval_task_name = self.task_names[eval_task_idx] if eval_task_idx < len(self.task_names) else f"task_{eval_task_idx}"
                
                # 获取评估数据加载器
                eval_task_info = continual_dataset.get_task_by_id(eval_task_idx, split='validation')
                if eval_task_info:
                    _, eval_loader, _ = eval_task_info
                    eval_acc = self.evaluate(eval_loader, eval_task_idx)
                    
                    # 记录性能到遗忘评估器
                    stage_name = f"after_task_{task_id}"
                    self.forgetting_evaluator.record_task_performance_by_name(
                        eval_task_name, stage_name, eval_acc
                    )
                    
                    print(f"  {eval_task_name} 在阶段 '{stage_name}' 的准确率: {eval_acc:.4f}")
            
            # 计算当前的遗忘指标
            current_forgetting = self.forgetting_evaluator.compute_average_forgetting()
            current_bwt = self.forgetting_evaluator.compute_backward_transfer(self.task_names[:task_id+1])
            
            print(f"当前平均遗忘率: {current_forgetting:.4f}")
            print(f"当前向后迁移: {current_bwt:.4f}")
        
        print("\n持续学习训练完成！")
        
        # 计算最终指标
        final_metrics = self.forgetting_evaluator.get_comprehensive_metrics(self.task_names)
        
        # 保存训练历史
        self.save_training_history()
        
        return final_metrics
    
    def save_checkpoint(self, task_id: int):
        """保存检查点"""
        checkpoint_path = f"{self.output_dir}/checkpoints/model_task_{task_id}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'training_history': self.training_history,
            'current_task_id': self.current_task_id,
            'task_performance_history': self.task_performance_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存到: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.config = checkpoint['config']
        self.training_history = checkpoint['training_history']
        self.current_task_id = checkpoint['current_task_id']
        self.task_performance_history = checkpoint['task_performance_history']
        
        print(f"检查点已从 {checkpoint_path} 加载")
    
    def save_training_history(self):
        """获取最终摘要并保存完整的训练历史"""
        history_path = f"{self.output_dir}/training_history.json"
        
        # 1. Get the final summary with all calculated metrics
        final_summary = self.get_training_summary()
        
        # 2. Merge the summary into the main training history object
        # This ensures that calculated fields like 'average_forgetting' are saved.
        self.training_history.update(final_summary)
        
        # 3. Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, dict):
                serializable_history[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                           for k, v in value.items()}
            else:
                serializable_history[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        # 4. Save the merged and serialized history to a file
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, ensure_ascii=False, indent=4)
        
        print(f"训练历史已保存到: {history_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        summary = {
            'total_tasks': len(self.task_performance_history),
            'final_accuracies': {},
            'average_forgetting': 0.0,
            'average_backward_transfer': 0.0, # BWT is complex, keeping it simple for now
            'forgetting_rates': {}
        }
        
        # 最终准确率
        for task_id, performances in self.task_performance_history.items():
            if performances:
                summary['final_accuracies'][task_id] = performances[-1]

        # 计算遗忘率
        all_forgetting_rates = []
        for task_id, performances in self.task_performance_history.items():
            # Can only calculate forgetting if a task has been evaluated more than once
            if len(performances) >= 2:
                # Max accuracy before the final one
                peak_performance = max(performances[:-1])
                final_performance = performances[-1]
                
                forgetting = max(0.0, peak_performance - final_performance)
                summary['forgetting_rates'][task_id] = forgetting
                all_forgetting_rates.append(forgetting)
        
        if all_forgetting_rates:
            summary['average_forgetting'] = np.mean(all_forgetting_rates)
        
        return summary