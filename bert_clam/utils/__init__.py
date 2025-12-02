"""
BERT-CLAM工具函数模块
"""

from .logger import setup_logger, get_logger
from .config_loader import load_config

__all__ = [
    "setup_logger",
    "get_logger",
    "load_config"
]