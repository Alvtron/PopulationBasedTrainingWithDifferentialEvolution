import os
from abc import abstractmethod
from typing import Any

import torch


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

    def __call__(self, device: str, argument: Any) -> Any:
        if not device.startswith('cuda'):
            return self.function(device, argument)
        with torch.cuda.device(device):
            return self.function(device, argument)