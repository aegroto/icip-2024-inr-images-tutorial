from typing import Callable
import copy
import statistics

from dataclasses import dataclass

from torch import Tensor, nn
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
    shuffle_factor: int = 1
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

        self.__config = config
        self.__logger = init_logger(config.name)

        self.__device = device or load_device()

        self.__target_image = target_image.to(self.__device)

    def __shuffle_tensor(self, tensor: Tensor) -> Tensor:
        unsqueezed = tensor.movedim(2, 0).unsqueeze(1)
        shuffled = nn.functional.pixel_unshuffle(
            unsqueezed, self.__config.shuffle_factor
        )
        reshaped = shuffled.movedim(0, 3)
        return reshaped

    def __generate_batch(self) -> TrainingBatch:
        reshaped_image = self.__target_image.movedim(0, 2)

        input = self.__model.generate_input(reshaped_image.shape).to(self.__device)

        self.__logger.debug(
            f"Unshuffled input and image shapes: {input.shape} {reshaped_image.shape}"
        )

        shuffled_input = self.__shuffle_tensor(input)
        shuffled_image = self.__shuffle_tensor(reshaped_image)

        self.__logger.debug(
            f"Shuffled input and image shapes: {shuffled_input.shape} {shuffled_image.shape}"
        )

        batch = TrainingBatch()

        for input_sample, image_sample in zip(
            shuffled_input.unbind(0), shuffled_image.unbind(0)
        ):
            self.__logger.debug(
                f"Sample input and image shapes: {input_sample.shape} {image_sample.shape}"
            )
            batch.add_sample(TrainingSample(input_sample, image_sample))

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
        scheduler = self.__config.scheduler_builder(optimizer)
        loss_fn = self.__config.loss_fn_builder()

        self.__logger.debug(
            f"Training setup: \nOptimizer: {optimizer}\n Scheduler: {scheduler}\n Loss function {loss_fn}"
        )

        for iteration in range(1, self.__config.iterations + 1):
            self.__current_iterations += 1
            self.__log(
                f"Iteration #{iteration}/{self.__config.iterations}:: Learning rate: {scheduler.get_last_lr()}"
            )

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
            scheduler.step()
