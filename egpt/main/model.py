from dataclasses import dataclass, field
from .training import Training
import tensorflow as tf

@dataclass(frozen=False, unsafe_hash=True)
class Model(Training):
   pass