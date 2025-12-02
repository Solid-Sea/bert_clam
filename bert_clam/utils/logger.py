"""
BERT-CLAM日志工具
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = 'bert_clam', 
                log_file: Optional[str] = None, 
                level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 文件处理器
    if log_file is None:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/bert_clam_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'bert_clam') -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)


# 全局日志记录器
logger = setup_logger()


def log_task_info(task_id: int, task_name: str, metrics: dict):
    """记录任务信息"""
    logger.info(f"Task {task_id} ({task_name}) completed:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


def log_training_epoch(epoch: int, loss: float, accuracy: float):
    """记录训练轮次信息"""
    logger.info(f"Epoch {epoch} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


def log_evaluation_results(results: dict):
    """记录评估结果"""
    logger.info("Evaluation Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")