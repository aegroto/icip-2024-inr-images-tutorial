import torch

from config import FittingPhaseConfiguration
from modules.nn.image_representation.coordinates_based import (
    CoordinatesBasedRepresentation,
)
from modules.nn.positional_encoder import PositionalEncoder
from modules.nn.quantizer.uniform import UniformQuantizer
from modules.nn.siren import Siren
from modules.training import TrainerConfiguration


def model_builder():
    encoder = PositionalEncoder(num_frequencies=16, scale=1.4)

    network = Siren(
        input_features=encoder.output_features_for(2),
        hidden_features=128,
        hidden_layers=2,
        output_features=3,
    )

    return CoordinatesBasedRepresentation(encoder, network)


def quantizer_builder(_):
    return UniformQuantizer(8)


def optimizer_builder(parameters):
    return torch.optim.Adam(parameters, lr=1.0e-3)


def scheduler_builder(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 1.0e-4)


def loss_fn_builder():
    return torch.nn.MSELoss()


phases = {
    "full_precision": FittingPhaseConfiguration(
        model_builder=model_builder,
        trainer_configuration=TrainerConfiguration(
            optimizer_builder=optimizer_builder,
            scheduler_builder=scheduler_builder,
            loss_fn_builder=loss_fn_builder,
            iterations=100,
            log_interval=10,
        ),
    ),
    "8bits_qat": FittingPhaseConfiguration(
        model_builder=model_builder,
        trainer_configuration=TrainerConfiguration(
            optimizer_builder=optimizer_builder,
            scheduler_builder=scheduler_builder,
            loss_fn_builder=loss_fn_builder,
            iterations=50,
            log_interval=10,
        ),
        quantizer_builder=quantizer_builder,
        recalibrate_quantizers=True,
    ),
}
