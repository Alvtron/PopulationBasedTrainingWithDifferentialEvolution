import os
from abc import abstractmethod
from typing import Any

import torch

__global_device = 'cpu'

def get_global_device() -> str:
    return __global_device

def set_global_device(device: str) -> None:
    global __global_device
    __global_device = device

def initialize_cuda_device(device: str) -> None:
    if not torch.cuda.is_available():
        raise Exception("cannot initialize CUDA because is not available")
    if not device.startswith('cuda'):
        raise Exception(f"cannot initialize CUDA with the provided device '{device}'")
    torch.cuda.set_device(device)
    torch.cuda.init()

class DeviceCallable(object):
    def __init__(self, verbose: bool = False):
        if not isinstance(verbose, bool):
            raise TypeError(f"the 'verbose' specified was of wrong type {type(verbose)}, expected {bool}.")
        self.verbose = verbose

    def _print(self, message: str):
        if not self.verbose:
            return
        full_message = f"PID-{os.getpid()}: {message}"
        print(full_message)

    @abstractmethod
    def function(self, device: str, argument: Any) -> Any:
        raise NotImplementedError()

    def __call__(self, argument: Any) -> Any:
        device = get_global_device()
        if not device.startswith('cuda'):
            return self.function(device, argument)
        with torch.cuda.device(device):
            return self.function(device, argument)