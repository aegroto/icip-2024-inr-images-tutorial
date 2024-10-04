from typing import Tuple
import constriction
import struct
import numpy
import torch
import math

from torch import Tensor
from modules.logging import init_logger


LOGGER = init_logger(__name__)


def build_range_encoder():
    return constriction.stream.queue.RangeEncoder()


def build_range_decoder(compressed_buffer):
    return constriction.stream.queue.RangeDecoder(compressed_buffer)


def build_laplace_entropy_model(min_symbol, max_symbol, mean, std):
    scale = std / math.sqrt(2.0)
    return constriction.stream.model.QuantizedLaplace(
        min_symbol, max_symbol, mean, scale
    )


def entropy_encode(
    quantized_tensor: Tensor, encoder_builder, entropy_model_builder
) -> bytes:
    symbols = quantized_tensor.cpu().to(torch.int32).numpy().flatten()
    num_symbols = len(symbols)

    LOGGER.debug(f"Encoder symbols head: {symbols[0:16]}")
    LOGGER.debug(f"Encoder symbols tail: {symbols[-16:]}")

    min_symbol = numpy.min(symbols)
    max_symbol = numpy.max(symbols)
    mean = float(numpy.mean(symbols).astype(numpy.float32))
    std = float(numpy.std(symbols).astype(numpy.float32))

    entropy_model = entropy_model_builder(min_symbol, max_symbol, mean, std)

    encoder = encoder_builder()
    encoder.encode(symbols, entropy_model)

    compressed_symbols = encoder.get_compressed()

    LOGGER.debug(f"Encoder compressed symbols head: {compressed_symbols[0:16]}")
    LOGGER.debug(f"Encoder compressed symbols tail: {compressed_symbols[-16:]}")

    compressed_buffer = compressed_symbols.tobytes()
    compressed_buffer_len = len(compressed_buffer)

    stream = bytes()

    LOGGER.debug(f"Saving entropy model parameters:: min_symbol: {min_symbol}, max_symbol: {max_symbol}, mean: {mean}, std: {std}")


    LOGGER.debug(f"Encoder:: num symbols: {num_symbols}, compressed_buffer_len: {compressed_buffer_len}")

    stream += struct.pack("!i", min_symbol)
    stream += struct.pack("!i", max_symbol)
    stream += struct.pack("!f", mean)
    stream += struct.pack("!f", std)
    stream += struct.pack("!I", num_symbols)

    stream += struct.pack("!I", compressed_buffer_len)
    stream += compressed_buffer

    return stream


def entropy_decode(
    compressed_stream: bytes, decoder_builder, entropy_model_builder
) -> Tuple[Tensor, int]:
    read_bytes = 0

    min_symbol = struct.unpack("!i", compressed_stream[read_bytes : read_bytes + 4])[0]
    read_bytes += 4
    max_symbol = struct.unpack("!i", compressed_stream[read_bytes : read_bytes + 4])[0]
    read_bytes += 4
    mean = struct.unpack("!f", compressed_stream[read_bytes : read_bytes + 4])[0]
    read_bytes += 4
    std = struct.unpack("!f", compressed_stream[read_bytes : read_bytes + 4])[0]
    read_bytes += 4

    LOGGER.debug(f"Unpacked entropy model parameters:: min_symbol: {min_symbol}, max_symbol: {max_symbol}, mean: {mean}, std: {std}")

    num_symbols = struct.unpack("!I", compressed_stream[read_bytes : read_bytes + 4])[0]
    read_bytes += 4

    compressed_buffer_len = struct.unpack(
        "!I", compressed_stream[read_bytes : read_bytes + 4]
    )[0]
    read_bytes += 4

    LOGGER.debug(f"Decoder:: num symbols: {num_symbols}, compressed_buffer_len: {compressed_buffer_len}")

    compressed_buffer = compressed_stream[
        read_bytes : read_bytes + compressed_buffer_len
    ]
    read_bytes += compressed_buffer_len
    compressed_symbols = numpy.frombuffer(compressed_buffer, numpy.uint32)

    LOGGER.debug(f"Decoder compressed symbols head: {compressed_symbols[0:16]}")
    LOGGER.debug(f"Decoder compressed symbols tail: {compressed_symbols[-16:]}")

    entropy_model = entropy_model_builder(min_symbol, max_symbol, mean, std)

    decoder = decoder_builder(compressed_symbols)
    symbols = decoder.decode(entropy_model, num_symbols)

    LOGGER.debug(f"Decoder symbols head: {symbols[0:16]}")
    LOGGER.debug(f"Decoder symbols tail: {symbols[-16:]}")

    return (torch.from_numpy(symbols).to(torch.float32), read_bytes)
