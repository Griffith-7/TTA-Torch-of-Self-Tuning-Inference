"""TTA-Torch: Dynamic Test-Time Adaptation for LLMs.

Real-time weight updates during generation via LoRA adapters.
The model adjusts its own parameters mid-generation based on its confidence level.

CLI: tta-torch generate | benchmark | clean
"""

import logging

from . import engine, loader
from .engine import TTAModel
from .loader import load_tta_model

__all__ = [
    "TTAModel",
    "load_tta_model",
    "engine",
    "loader",
]

__version__ = "0.3.0"

# Configure default logger
logging.getLogger(__name__).addHandler(logging.NullHandler())
