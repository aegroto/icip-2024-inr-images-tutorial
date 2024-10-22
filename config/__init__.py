from dataclasses import dataclass
from typing import Callable


@dataclass
class FittingPhaseConfiguration:
    model_builder: Callable
    trainer_builder: Callable
    recalibrate_quantizers: bool = False
    quantizer_builder: Callable = None
