from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SamplingParams:
    """Parameters controlling token sampling during generation."""
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    frequency_penalty: float = 0.2
    eos_token_id: Optional[int] = None
    exclude_token_ids: List[int] = field(default_factory=list)
    max_new_tokens: int = 20
