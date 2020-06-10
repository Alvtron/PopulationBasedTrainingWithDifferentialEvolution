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

    times=5

    tasks=(
        'fashionmnist_lenet5',
        'mnist_lenet5',
        'fashionmnist_mlp',
        'mnist_mlp')

    evolvers=(
        'pbt',
        'de',
        'shade',
        'lshade')

    n_tasks = times * len(tasks) * len(evolvers)

    for i in range(times):
        for task in tasks:
            for evolver in evolvers:
                print(f"({i} of {n_tasks}) running '{task}' with '{evolver}'...")
                run(task=task, evolver=evolver, population_size=30, batch_size=64,
                    train_steps=250, fitness_steps=0, end_steps=30*40,
                    n_jobs=8, devices=['cuda:0'], tensorboard=False,
                    verbose=2, logging=True)