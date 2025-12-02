"""
BERT-CLAM核心模块
"""

from .amr import AdaptiveMemoryRetrieval
from .ewc import ElasticWeightConsolidation
from .alp import AdaptiveLoRAFusion
from .grammar_aware import GrammarAwareModule

__all__ = [
    "AdaptiveMemoryRetrieval",
    "ElasticWeightConsolidation", 
    "AdaptiveLoRAFusion",
    "GrammarAwareModule"
]