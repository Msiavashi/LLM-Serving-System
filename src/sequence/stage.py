from enum import Enum


class Stage(str, Enum):
    """Sequence execution stage in the inference pipeline."""
    PREFILL = "prefill"
    DECODE = "decode"
