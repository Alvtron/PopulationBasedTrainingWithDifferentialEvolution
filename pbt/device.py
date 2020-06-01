import os
from abc import abstractmethod

class DeviceCallable(object):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _print(self, message: str):
        if not self.verbose:
            return
        full_message = f"PID-{os.getpid()}: {message}"
        print(full_message)

    @abstractmethod
    def __call__(self, checkpoint: object, device: str, **kwargs) -> object:
        raise NotImplementedError