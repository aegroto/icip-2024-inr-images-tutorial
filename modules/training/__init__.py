from typing import Callable
import torch
import copy
import statistics

from dataclasses import dataclass, field

from torch import nn, Tensor
from modules.device import load_device
from modules.logging import init_logger
from modules.nn.image_representation.base import ImplicitImageRepresentation
from modules.training.batch import TrainingBatch, TrainingSample


@dataclass
class TrainerConfiguration:
    iterations: int
    optimizer_builder: Callable
    scheduler_builder: Callable
    loss_fn_builder: Callable
    log_interval: int = 1
    name: str = __name__


class Trainer:
    def __init__(
        self,
        config: TrainerConfiguration,
        model: ImplicitImageRepresentation,
        target_image: Tensor,
        device=None,
    ) -> None:
        self.__model = model
        self.__target_image = target_image

        self.__config = config
        self.__logger = init_logger(config.name)

        self.__device = device or load_device()

    def __generate_batch(self) -> TrainingBatch:
        reshaped_target_image = self.__target_image.movedim(0, 2).to(self.__device)

        input = self.__model.generate_input(reshaped_target_image.shape).to(
            self.__device
        )

        batch = TrainingBatch()
        batch.add_sample(TrainingSample(input, reshaped_target_image))

        return batch

    def __log(self, content):
        if self.__current_iterations % self.__config.log_interval != 0:
            return

        self.__logger.info(content)

    def best_result(self):
        return (self.__best_model, self.__best_loss_value)

    def train(self):
        batch = self.__generate_batch()
        model = self.__model

        self.__best_loss_value = float("inf")
        self.__best_model = copy.deepcopy(model)

        self.__current_iterations = 0

        optimizer = self.__config.optimizer_builder(model.parameters())
        loss_fn = self.__config.loss_fn_builder()

        self.__logger.debug(f"Training with optimizer: {optimizer}")
        self.__logger.debug(f"Training with loss function: {loss_fn}")

        for iteration in range(1, self.__config.iterations + 1):
            self.__current_iterations += 1
            self.__log(f"Iteration #{iteration}/{self.__config.iterations}")

            loss_norm = 1.0 / batch.size()

            optimizer.zero_grad()

            iteration_loss_values = list()

            for input, target_output in batch.samples():
                reconstructed_output = model(input)

                loss_value = loss_fn(target_output, reconstructed_output)

                iteration_loss_values.append(loss_value.item())

                loss_value *= loss_norm
                loss_value.backward()

            average_loss_value = statistics.mean(iteration_loss_values)

            self.__log(f"Average iteration loss value: {average_loss_value:.5f}")

            if average_loss_value < self.__best_loss_value:
                self.__best_loss_value = average_loss_value
                self.__best_model = copy.deepcopy(model)

            optimizer.step()
