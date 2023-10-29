from dataclasses import dataclass, field
import tensorflow as tf
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Training:
    pass