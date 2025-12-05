import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import logging
import os
import sys
import json
import argparse
from pathlib import Path

# 将项目根目录添加到路径中，以便直接执行脚本
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, project_root)

from bert_clam.models.bert_clam_model import BERTCLAMModel
from bert_clam.training.trainer import BERTCLAMTrainer
from bert_clam.core.strategy import (
    ContinualLearningStrategy,
    EWCStrategy,
    MRBStrategy,
    ALPStrategy,
    GrammarStrategy
)

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 策略工厂 ---
def create_strategies_from_config(config, model):
    """
    根据配置创建策略列表
    
    Args:
        config: 实验配置字典
        model: BERTCLAMModel 实例
        
    Returns:
        strategies: 策略对象列表
    """
    strategies = []
    
    # 检查是否有新的 strategies 配置
    if "strategies" in config:
        for strategy_config in config["strategies"]:
            if not strategy_config.get("enabled", True):
                continue
            
            strategy_type = strategy_config["type"]
            params = strategy_config.get("params", {})
            
            if strategy_type == "grammar" and model.grammar_aware:
                strategies.append(GrammarStrategy(model.grammar_aware))
            elif strategy_type == "alp" and model.alp:
                strategies.append(ALPStrategy(model.alp))
            elif strategy_type == "mrb" and model.mrb:
                fusion_weight = params.get("fusion_weight", 0.2)
                strategies.append(MRBStrategy(model.mrb, fusion_weight))
            elif strategy_type == "ewc" and model.ewc:
                strategies.append(EWCStrategy(model.ewc, model))
    
    return strategies

# --- 数据准备 ---
def prepare_dataset(task_name, tokenizer, max_length, num_train_samples=1000, num_val_samples=500):
    """根据任务名称加载并准备数据集"""
    logging.info(f"为任务加载和准备数据集: {task_name}")
    
    try:
        if task_name == "mnli":
            dataset = load_dataset("glue", "mnli")
            train_dataset = dataset["train"].shuffle(seed=42).select(range(num_train_samples))
            val_dataset = dataset["validation_matched"].shuffle(seed=42).select(range(num_val_samples))
            
            def encode(examples):
                return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=max_length)
            
        elif task_name in ["qnli", "qqp"]:
            dataset = load_dataset("glue", task_name)
            train_dataset = dataset["train"].shuffle(seed=42).select(range(num_train_samples))
            val_dataset = dataset["validation"].shuffle(seed=42).select(range(num_val_samples))
            
            text_keys = ('question', 'sentence') if task_name == "qnli" else ('question1', 'question2')
            
            def encode(examples):
                return tokenizer(examples[text_keys[0]], examples[text_keys[1]], truncation=True, padding='max_length', max_length=max_length)
        else:
            raise ValueError(f"不支持的任务: {task_name}")

        train_dataset = train_dataset.map(encode, batched=True)
        val_dataset = val_dataset.map(encode, batched=True)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        return train_dataset, val_dataset
    except Exception as e:
        logging.error(f"加载任务 {task_name} 数据时出错: {e}")
        logging.info("请确保您已连接到互联网以下载GLUE数据集。")
        sys.exit(1)

# --- 主实验脚本 ---
def run_experiment(config):
    """
    根据提供的配置运行单个持续学习实验。
    """
    # 1. 创建输出目录
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"实验结果将保存在: {output_dir}")

    # 2. 初始化 Tokenizer 和数据整理器
    tokenizer = AutoTokenizer.from_pretrained(config.get("bert_model", "bert-base-uncased"))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. 初始化模型
    # 确定所有任务中最大的标签数，以设置分类器头
    num_labels_per_task = {"mnli": 3, "qnli": 2, "qqp": 2}
    max_labels = max(num_labels_per_task.values())
    
    model_config = config.get("model_config", {})
    logging.info(f"使用模型配置初始化模型: {model_config}")
    
    model = BERTCLAMModel(
        model_name=config.get("bert_model", "bert-base-uncased"),
        num_labels=max_labels,
        # 从配置中获取模块开关，默认为False
        enable_ewc=model_config.get("enable_ewc", False),
        enable_amr=model_config.get("enable_amr", False),
        enable_alp=model_config.get("enable_alp", False),
        enable_grammar=model_config.get("enable_grammar", False),
        # 从配置中获取模块特定参数
        ewc_lambda=model_config.get("ewc_lambda", 0.0),
        amr_k=model_config.get("amr_k", 5),
        alp_top_k=model_config.get("alp_top_k", 3),
        # 基线实验：禁用LoRA以进行标准Fine-tuning
        lora_enabled=model_config.get("lora_enabled", False)
    ).to(DEVICE)
    
    # 如果配置中有策略定义，创建并注入策略
    strategies = create_strategies_from_config(config, model)
    if strategies:
        model.strategies = strategies
        logging.info(f"已加载 {len(strategies)} 个策略: {[s.name for s in strategies]}")

    # 4. 初始化训练器
    training_config = config.get("training_config", {})
    use_wandb = config.get("use_wandb", True)
    wandb_project = config.get("wandb_project", "bert-clam")
    experiment_name = config.get("experiment_name", output_dir.name)
    
    trainer = BERTCLAMTrainer(
        model=model,
        config=training_config,
        output_dir=str(output_dir),
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=experiment_name
    )
    
    # 5. 持续学习循环
    tasks = config["tasks"]
    all_task_datasets = {}
    
    for i, task_name in enumerate(tasks):
        task_id = i
        logging.info(f"\n{'='*50}\n>>> 开始任务 {task_id + 1}/{len(tasks)}: {task_name.upper()} <<<\n{'='*50}")

        # 准备当前任务的数据
        train_dataset, val_dataset = prepare_dataset(
            task_name, 
            tokenizer, 
            max_length=training_config.get("max_length", 128),
            num_train_samples=training_config.get("num_train_samples", 1000),
            num_val_samples=training_config.get("num_val_samples", 500)
        )
        all_task_datasets[task_name] = (train_dataset, val_dataset)

        # 训练当前任务
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_config.get("batch_size", 16), collate_fn=data_collator, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=training_config.get("batch_size", 16), collate_fn=data_collator)
        
        trainer.train_task(
            task_id=task_id, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            epochs=training_config.get("epochs_per_task", 1)
        )

        # 评估阶段：在所有已见过的任务上进行评估
        logging.info(f"--- 在训练 {task_name.upper()} 后评估性能 ---")
        
        # 保存当前模型状态（训练完当前任务后的状态）
        current_model_state = {
            'model': model.state_dict(),
            'optimizer': trainer.optimizer.state_dict()
        }
        
        for j, past_task_name in enumerate(tasks[:i+1]):
            past_task_id = j
            _, past_val_dataset = all_task_datasets[past_task_name]
            past_val_loader = torch.utils.data.DataLoader(past_val_dataset, batch_size=training_config.get("batch_size", 16), collate_fn=data_collator)
            
            # 使用当前模型状态评估（训练完当前任务后的状态）
            accuracy = trainer.evaluate(past_val_loader, task_id=past_task_id, record_performance=True)
            logging.info(f"  - 在 '{past_task_name}' 上的准确率: {accuracy:.4f}")
        
        # 恢复当前模型状态
        model.load_state_dict(current_model_state['model'])
        trainer.optimizer.load_state_dict(current_model_state['optimizer'])
        logging.info(f"[FIX] 已恢复到当前任务 {task_id} 的模型状态")

    # 6. 最终分析和结果保存
    logging.info(f"\n{'='*50}\n>>> 最终遗忘分析 <<<\n{'='*50}")
    final_summary = trainer.get_training_summary()
    
    logging.info("来自训练器的最终摘要:")
    for key, value in final_summary.items():
        if isinstance(value, dict):
            logging.info(f"  - {key}:")
            for sub_key, sub_value in value.items():
                logging.info(f"    - {sub_key}: {sub_value:.4f}")
        else:
            logging.info(f"  - {key}: {value:.4f}")
            
    trainer.save_training_history()
    logging.info(f"\n训练历史已保存到 {trainer.output_dir}/training_history.json")
    logging.info("\n实验成功完成!")

def main():
    """主函数：解析参数并启动实验"""
    parser = argparse.ArgumentParser(description="运行基于JSON配置的BERT-CLAM持续学习实验。")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="实验配置文件的路径 (JSON格式)。"
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"配置文件未找到: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"解析JSON配置文件时出错: {args.config}")
        sys.exit(1)

    run_experiment(config)

if __name__ == "__main__":
    main()