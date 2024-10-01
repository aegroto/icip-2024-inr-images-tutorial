import torch

from modules.nn.image_representation.siren import SirenRepresentation
from modules.nn.positional_encoder import PositionalEncoderConfig
from modules.nn.siren import SirenConfig
from modules.training import TrainerConfiguration

model = SirenRepresentation(
    encoder_config=PositionalEncoderConfig(num_frequencies=16, scale=1.4),
    network_config=SirenConfig(
        input_features=2,
        hidden_features=128,
        hidden_layers=2,
        output_features=3,
    ),
)

trainer_configuration = TrainerConfiguration(
    iterations=100, optimizer_parameters={"lr": 1.0e-2}
)
