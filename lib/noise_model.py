from typing import Union
from tequila.circuit.noise import NoiseModel, BitFlip, DepolarizingError

__all__ = ['get_noise_model', 'noise_types']

noise_types = ['bitflip-depol', 'device', None]


def get_noise_model(error_rate: float = 1e-2, noise_type: str = None) -> Union[NoiseModel, str]:
    try:
        return {'bitflip-depol': BitFlip(p=error_rate, level=1) + DepolarizingError(p=error_rate, level=2),
                'device': 'device',
                None: None}[noise_type]
    except KeyError:
        raise KeyError(f'unknown noise_type! got {noise_type}, must be one of {noise_types}')
