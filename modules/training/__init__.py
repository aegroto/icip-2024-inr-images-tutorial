import torch
import statistics

from dataclasses import dataclass, field

from torch import nn, Tensor
from modules.logging import init_logger
from modules.nn.base import ImplicitImageRepresentation
from modules.training.batch import TrainingBatch, TrainingSample

@dataclass
class TrainerConfiguration:
    iterations: int
    optimizer_parameters: dict = field(default_factory=dict)
    name: str = __name__

class Trainer:
    def __init__(self, config: TrainerConfiguration, model: ImplicitImageRepresentation, target_image: Tensor) -> None:
        self.__model = model
        self.__target_image = target_image

        self.__config = config
        self.__logger = init_logger(config.name)


    def __generate_batch(self) -> TrainingBatch:
        reshaped_target_image = self.__target_image.movedim(0, 2)

        input = self.__model.generate_input(reshaped_target_image.shape)

        batch = TrainingBatch()
        batch.add_sample(TrainingSample(input, reshaped_target_image))

        return batch

    def train(self):
        batch = self.__generate_batch()
        model = self.__model

        optimizer = torch.optim.Adam(model.parameters(), **self.__config.optimizer_parameters)

        loss_fn = nn.MSELoss()

        for iteration in range(1, self.__config.iterations+1):
            self.__logger.debug(f"Iteration #{iteration}")

            loss_norm = 1.0 / batch.size()

            optimizer.zero_grad()

            iteration_loss_values = list()

            for (input, target_output) in batch.samples():
                reconstructed_output = model(input)
                
                loss_value = loss_fn(target_output, reconstructed_output) 

                iteration_loss_values.append(loss_value.item())
                
                loss_value *= loss_norm
                loss_value.backward()

            average_loss_value = statistics.mean(iteration_loss_values)
            self.__logger.debug(f"Average iteration loss value: {average_loss_value:.5f}") 

            optimizer.step()


