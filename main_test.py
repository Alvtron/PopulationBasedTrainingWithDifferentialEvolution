import random

import torch
import numpy as np

from main_helper import run

if __name__ == "__main__":
    # set multiprocessing settings
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    # set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    # set torch settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    run(task='fashionmnist_lenet5', evolver='lshade', population_size=6, batch_size=64,
        train_steps=242, fitness_steps=8, end_steps=30 * 40, n_jobs=2, devices=['cuda:0'], tensorboard=False, verbose=2, logging=True)