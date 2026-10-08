"""Prompt registry for the EU Taxonomy RAG pipeline.

Each module contains one prompt level (basic, intermediate, expert).
Import PROMPTS to access them by name, or import individual prompts directly.
"""

from .basic import BASIC_PROMPT
from .intermediate import INTERMEDIATE_PROMPT
from .expert import EXPERT_PROMPT

PROMPTS = {
    "basic": BASIC_PROMPT,
    "intermediate": INTERMEDIATE_PROMPT,
    "expert": EXPERT_PROMPT,
}
