import torch

from modules.nn.image_representation.coordinates_based import CoordinatesBasedRepresentation
from modules.nn.mlp import MultiLayerPerceptronConfig
from modules.nn.positional_encoder import PositionalEncoderConfig
from modules.training import TrainerConfiguration

model = CoordinatesBasedRepresentation(
        encoder_config=PositionalEncoderConfig(num_frequencies=16, scale=1.4),
        network_config=MultiLayerPerceptronConfig(
            input_features=2,
            hidden_features=128,
            hidden_layers=3,
            output_features=3,
            activation_builder=lambda: torch.nn.GELU(),
        ),
    )

trainer_configuration = TrainerConfiguration(iterations=100, optimizer_parameters={
        "lr": 1.0e-2
    })
