from dataclasses import dataclass
from typing import List

from torch import Tensor


@dataclass
class TrainingSample:
    input: Tensor
    target_output: Tensor

    def as_tuple(self) -> tuple[Tensor, Tensor]:
        return (self.input, self.target_output)


class TrainingBatch:
    def __init__(self):
        self.__samples = list()

    def add_sample(self, sample: TrainingSample):
        self.__samples.append(sample)

    def samples(self) -> List[TrainingSample]:
        return [sample.as_tuple() for sample in self.__samples]

    def size(self) -> int:
        return len(self.__samples)
