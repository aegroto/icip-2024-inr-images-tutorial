from config import Configuration
from modules.nn.image_representation.siren import SirenRepresentation
from modules.nn.positional_encoder import PositionalEncoderConfig
from modules.nn.quantizer.uniform import UniformQuantizer
from modules.nn.siren import SirenConfig
from modules.training import TrainerConfiguration

model = SirenRepresentation(
    encoder_config=PositionalEncoderConfig(num_frequencies=16, scale=1.4),
    network_config=SirenConfig(
        input_features=2,
        hidden_features=256,
        hidden_layers=1,
        output_features=3,
    ),
)

quantizer_builder = lambda _: UniformQuantizer(8)

phases = {
    "fitting": Configuration(
        model=model,
        trainer_configuration=TrainerConfiguration(
            iterations=1000, optimizer_parameters={"lr": 1.0e-4}, log_interval=10
        ),
    ),
    "quantization": Configuration(
        model=model,
        trainer_configuration=TrainerConfiguration(
            iterations=100, optimizer_parameters={"lr": 1.0e-4}, log_interval=10
        ),
        quantizer_builder=quantizer_builder,
        recalibrate_quantizers=True
    ),
}
