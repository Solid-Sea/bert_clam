"""
BERT-CLAM核心模块
"""

from .memory_replay_bank import AdaptiveMemoryRetrieval
from .ewc import ElasticWeightConsolidation
from .alp import AdaptiveLoRAFusion
from .grammar_aware import GrammarAwareModule
from .strategy import (
    ContinualLearningStrategy,
    EWCStrategy,
    MRBStrategy,
    ALPStrategy,
    GrammarStrategy
)

__all__ = [
    "AdaptiveMemoryRetrieval",
    "ElasticWeightConsolidation",
    "AdaptiveLoRAFusion",
    "GrammarAwareModule",
    "ContinualLearningStrategy",
    "EWCStrategy",
    "MRBStrategy",
    "ALPStrategy",
    "GrammarStrategy"
]