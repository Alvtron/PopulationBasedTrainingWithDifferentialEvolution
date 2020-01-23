from dataclasses import dataclass
from member import Checkpoint

@dataclass
class Trial(object):
    checkpoint : Checkpoint
    train_data : list
    eval_data : list