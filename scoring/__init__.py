"""BRIDGE 评分模块：attention / VJP / proxy / marginal / pipeline。"""

from scoring.attention import AttentionRecorder, real_prompt_response_positions
from scoring.vjp import VJPComputer
from scoring.jvp import JVPComputer
from scoring.proxy import InteractionProxy
from scoring.marginal import MarginalScorer
from scoring.pipeline import ScoringPipeline, SampleRecord, run_scoring

__all__ = [
    "AttentionRecorder",
    "real_prompt_response_positions",
    "VJPComputer",
    "JVPComputer",
    "InteractionProxy",
    "MarginalScorer",
    "ScoringPipeline",
    "SampleRecord",
    "run_scoring",
]
