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

    population_size=10
    batch_size=64
    train_steps=242
    fitness_steps=8
    end_steps=30*40
    n_jobs=8
    devices=['cuda:0']
    tensorboard=False
    verbose=2
    logging=True

    tasks=(
        'fashionmnist_lenet5',
        #'mnist_lenet5',
        #'fashionmnist_mlp',
        #'mnist_mlp'
        )

    evolvers=(
        'pbt',
        'de',
        'shade',
        'lshade'
        )

    times=1

    n_tasks = times * len(tasks) * len(evolvers)
    n_done = 0

    for i in range(times):
        for task in tasks:
            for evolver in evolvers:
                if evolver == 'pbt':
                    t_steps = train_steps + fitness_steps
                    f_steps = 0
                else:
                    t_steps = train_steps
                    f_steps = fitness_steps
                print(f"({n_done + 1} of {n_tasks}) running '{task}' with '{evolver}'...")
                run(task=task, evolver=evolver, population_size=population_size, batch_size=batch_size,
                    train_steps=t_steps, fitness_steps=f_steps, end_steps=end_steps,
                    n_jobs=n_jobs, devices=devices, tensorboard=tensorboard,
                    verbose=verbose, logging=logging)
                n_done += 1