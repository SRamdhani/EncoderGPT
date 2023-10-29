from dataclasses import dataclass, field
from .encodergpt import EncoderDecoder
from .training import Training

@dataclass(frozen=False, unsafe_hash=True)
class Model(EncoderDecoder, Training):
   feed_foward_dim: int = field(init=True, default=int, repr=False, compare=False)
   vocab_size: int = field(init=True, default=int, repr=False, compare=False)
   seq_len: int = field(init=True, default=int, repr=False, compare=False)
   batch_size: int = field(init=True, default=int, repr=False, compare=False)
   embed_dim: int = field(init=True, default=int, repr=False, compare=False)
   num_heads: int = field(init=True, default=int, repr=False, compare=False)
   num_layers: int = field(init=True, default=int, repr=False, compare=False)

   def __post_init__(self) -> None:
      model_gpt, model_class = Model.sequentialEncoderDecoder(vocab_size=self.vocab_size,
                                                            seq_len=self.seq_len,
                                                            embed_dim=self.embed_dim,
                                                            num_layers=self.num_layers,
                                                            num_heads=self.num_heads,
                                                            feed_foward_dim=self.feed_foward_dim)

      object.__setattr__(self, 'model_gpt', model_gpt)
      object.__setattr__(self, 'model_class', model_class)

