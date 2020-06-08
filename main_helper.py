import os
import copy
import random
import time
import argparse
import multiprocessing
from typing import List, Sequence
from functools import partial
from dataclasses import dataclass

import torch
import numpy as np
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import pbt.evolution
from pbt.controller import PBTController, DEController
from pbt.task import mnist, emnist, fashionmnist, creditfraud, cifar
from pbt.analyze import Analyzer
from pbt.database import Database
from pbt.nn import Trainer, Evaluator, RandomFitnessApproximation


def import_task(task_name: str):
    # CREDIT CARD FRAUD
    if task_name == "creditfraud":
        return creditfraud.CreditCardFraud()
    # CIFAR 10
    elif task_name == "cifar10_mlp":
        return cifar.CIFAR10('mlp')
    elif task_name == "cifar10_lenet5":
        return cifar.CIFAR10('lenet5')
    elif task_name == "cifar10_vgg16":
        return cifar.CIFAR10('vgg16')
    elif task_name == "cifar10_resnet18":
        return cifar.CIFAR10('resnet18')
    # CIFAR 100
    elif task_name == "cifar100_mlp":
        return cifar.CIFAR100('mlp')
    elif task_name == "cifar100_lenet5":
        return cifar.CIFAR100('lenet5')
    elif task_name == "cifar100_vgg16":
        return cifar.CIFAR100('vgg16')
    elif task_name == "cifar100_resnet18":
        return cifar.CIFAR100('resnet18')
    # MNIST
    elif task_name == "mnist_mlp":
        return mnist.Mnist('mlp')
    elif task_name == "mnist_lenet5":
        return mnist.Mnist('lenet5')
    elif task_name == "mnist_vgg16":
        return mnist.Mnist('vgg16')
    elif task_name == "mnist_resnet18":
        return mnist.Mnist('resnet18')
    # EMNIST byclass
    elif task_name == "emnist_byclass_mlp":
        return emnist.EMnist('mlp', 'byclass')
    elif task_name == "emnist_byclass_lenet5":
        return emnist.EMnist('lenet5', 'byclass')
    elif task_name == "emnist_byclass_vgg16":
        return emnist.EMnist('vgg16', 'byclass')
    elif task_name == "emnist_byclass_resnet18":
        return emnist.EMnist('resnet18', 'byclass')
    # EMNIST bymerge
    elif task_name == "emnist_bymerge_mlp":
        return emnist.EMnist('mlp', 'bymerge')
    elif task_name == "emnist_bymerge_lenet5":
        return emnist.EMnist('lenet5', 'bymerge')
    elif task_name == "emnist_bymerge_vgg16":
        return emnist.EMnist('vgg16', 'bymerge')
    elif task_name == "emnist_bymerge_resnet18":
        return emnist.EMnist('resnet18', 'bymerge')
    # EMNIST balanced
    elif task_name == "emnist_balanced_mlp":
        return emnist.EMnist('mlp', 'balanced')
    elif task_name == "emnist_balanced_lenet5":
        return emnist.EMnist('lenet5', 'balanced')
    elif task_name == "emnist_balanced_vgg16":
        return emnist.EMnist('vgg16', 'balanced')
    elif task_name == "emnist_balanced_resnet18":
        return emnist.EMnist('resnet18', 'balanced')
    # EMNIST letters
    elif task_name == "emnist_letters_mlp":
        return emnist.EMnist('mlp', 'letters')
    elif task_name == "emnist_letters_lenet5":
        return emnist.EMnist('lenet5', 'letters')
    elif task_name == "emnist_letters_vgg16":
        return emnist.EMnist('vgg16', 'letters')
    elif task_name == "emnist_letters_resnet18":
        return emnist.EMnist('resnet18', 'letters')
    # EMNIST digits
    elif task_name == "emnist_digits_mlp":
        return emnist.EMnist('mlp', 'digits')
    elif task_name == "emnist_digits_lenet5":
        return emnist.EMnist('lenet5', 'digits')
    elif task_name == "emnist_digits_vgg16":
        return emnist.EMnist('vgg16', 'digits')
    elif task_name == "emnist_digits_resnet18":
        return emnist.EMnist('resnet18', 'digits')
    # EMNIST mnist
    elif task_name == "emnist_mnist_mlp":
        return emnist.EMnist('mlp', 'mnist')
    elif task_name == "emnist_mnist_lenet5":
        return emnist.EMnist('lenet5', 'mnist')
    elif task_name == "emnist_mnist_vgg16":
        return emnist.EMnist('vgg16', 'mnist')
    elif task_name == "emnist_mnist_resnet18":
        return emnist.EMnist('resnet18', 'mnist')
    # FashionMNIST
    elif task_name == "fashionmnist_mlp":
        return fashionmnist.FashionMnist('mlp')
    elif task_name == "fashionmnist_lenet5":
        return fashionmnist.FashionMnist('lenet5')
    elif task_name == "fashionmnist_vgg16":
        return fashionmnist.FashionMnist('vgg16')
    elif task_name == "fashionmnist_resnet18":
        return fashionmnist.FashionMnist('resnet18')
    else:
        raise NotImplementedError(
            f"Your requested task '{task_name}'' is not available.")


def create_evolver(manager, evolver_name, population_size, end_nfe): 
    if evolver_name == 'pbt':
        return pbt.evolution.ExploitAndExplore(
            exploit_factor=0.2, explore_factors=(0.8, 1.2))
    if evolver_name == 'de':
        return pbt.evolution.DifferentialEvolution(
            F=0.2, Cr=0.8)
    if evolver_name == 'shade':
        return pbt.evolution.SHADE(
            manager=manager, N_INIT=population_size,
            r_arc=2.0, p=0.2, memory_size=5)
    if evolver_name == 'lshade':
        return pbt.evolution.LSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5)
    if evolver_name == 'lshade_conservative':
        return pbt.evolution.LSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, f_min=0.0, f_max=0.5)
    if evolver_name == 'lshade_very_conservative':
        return pbt.evolution.LSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, f_min=0.0, f_max=0.1)
    if evolver_name == 'lshade_explorative':
        return pbt.evolution.LSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, f_min=0.0, f_max=2.0)
    if evolver_name == 'lshade_decay_linear':
        return pbt.evolution.DecayingLSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, decay_type='linear')
    if evolver_name == 'lshade_decay_curve':
        return pbt.evolution.DecayingLSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, decay_type='curve')
    if evolver_name == 'lshade_decay_logistic':
        return pbt.evolution.DecayingLSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, decay_type='logistic')
    if evolver_name == 'lshade_guide_linear':
        return pbt.evolution.GuidedLSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, guide_type='linear', strength=0.5)
    if evolver_name == 'lshade_guide_curve':
        return pbt.evolution.GuidedLSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, guide_type='curve', strength=0.5)
    if evolver_name == 'lshade_guide_logistic':
        return pbt.evolution.GuidedLSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, guide_type='logistic', strength=0.5)
    if evolver_name == 'lshade_state_sharing':
        return pbt.evolution.LSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, state_sharing=True)
    if evolver_name == 'lshade_state_sharing_conservative':
        return pbt.evolution.LSHADE(
            manager=manager, N_INIT=population_size, MAX_NFE=end_nfe,
            r_arc=2.0, p=0.2, memory_size=5, f_min=0.0, f_max=0.5, state_sharing=True)
    else:
        raise NotImplementedError(
            f"Your evolver request '{evolver_name}'' is not available.")


def create_tensorboard(log_directory):
    tensorboard_log_path = f"{log_directory}/tensorboard"
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tensorboard_log_path])
    url = tb.launch()
    return SummaryWriter(tensorboard_log_path), url


def run(
        task: str, evolver: str, population_size: int, batch_size: int,
        train_steps: int, fitness_steps: int = 0, end_steps: int = None,
        end_time: int = None, end_score: float = None, directory: str = 'checkpoints',
        devices: Sequence[str] = ['cpu'], n_jobs: int = 1, verbose: int = 1,
        logging: bool = True, history: int = 2, tensorboard: bool = False):
    if not isinstance(task, str):
        raise TypeError(f"the 'task' specified was of wrong type {type(task)}, expected {str}.")
    if not task:
        raise ValueError(f"the 'task' string specified was empty.")
    if not isinstance(evolver, str):
        raise TypeError(f"the 'evolver' specified was of wrong type {type(evolver)}, expected {str}.")
    if not evolver:
        raise ValueError(f"the 'evolver' string specified was empty.")
    if not isinstance(population_size, int):
        raise TypeError(f"the 'population_size' specified was of wrong type {type(population_size)}, expected {int}.")
    if population_size < 1:
        raise ValueError(f"the 'population_size' specified was less than 1.")
    if not isinstance(batch_size, int):
        raise TypeError(f"the 'batch_size' specified was of wrong type {type(batch_size)}, expected {int}.")
    if batch_size < 1:
        raise ValueError(f"the 'batch_size' specified was less than 1.")
    if not isinstance(train_steps, int):
        raise TypeError(f"the 'train_steps' specified was of wrong type {type(train_steps)}, expected {int}.")
    if train_steps < 1:
        raise ValueError(f"the 'train_steps' specified was less than 1.")
    if not isinstance(fitness_steps, int):
        raise TypeError(f"the 'fitness_steps' specified was of wrong type {type(fitness_steps)}, expected {int}.")
    if fitness_steps < 0:
        raise ValueError(f"the 'fitness_steps' specified was less than 0.")
    if end_steps is not None and not isinstance(end_steps, int):
        raise TypeError(f"the 'end_steps' specified was of wrong type {type(end_steps)}, expected {int}.")
    if end_time is not None and not isinstance(end_time, int):
        raise TypeError(f"the 'end_time' specified was of wrong type {type(end_time)}, expected {int}.")
    if end_score is not None and not isinstance(end_score, int):
        raise TypeError(f"the 'end_score' specified was of wrong type {type(end_score)}, expected {int}.")
    if not isinstance(directory, str):
        raise TypeError(f"the 'directory' path specified was of wrong type {type(directory)}, expected {str}.")
    if not directory:
        raise ValueError(f"the 'directory' path specified was empty.")
    if not isinstance(devices, (list, tuple)):
        raise TypeError(f"the 'devices' specified was of wrong type {type(devices)}, expected {list} or {tuple}.")
    if not isinstance(n_jobs, int):
        raise TypeError(f"the 'n_jobs' specified was of wrong type {type(n_jobs)}, expected {int}.")
    if n_jobs < 1:
        raise ValueError(f"the 'n_jobs' specified was less than 1.")
    if not isinstance(verbose, int):
        raise TypeError(f"the 'verbose' specified was of wrong type {type(verbose)}, expected {int}.")
    if not isinstance(logging, bool):
        raise TypeError(f"the 'logging' specified was of wrong type {type(logging)}, expected {bool}.")
    if not isinstance(history, int):
        raise TypeError(f"the 'history' specified was of wrong type {type(history)}, expected {int}.")
    if history < 1:
        raise ValueError(f"the 'history' specified was less than 1.")
    if not isinstance(tensorboard, bool):
        raise TypeError(f"the 'tensorboard' specified was of wrong type {type(tensorboard)}, expected {bool}.")
    # prepare objective
    print(f"Importing task...")
    _task = import_task(task)
    # create memory manager
    manager = torch.multiprocessing.Manager()
    # prepare database
    print(f"Preparing database...")
    database = Database(
        directory_path=f"{directory}/{task}_p{population_size}_train{train_steps}_fitness{fitness_steps}_batch{batch_size}_{evolver}",
        read_function=torch.load, write_function=torch.save)
    # prepare tensorboard writer
    tensorboard_writer = None
    if tensorboard:
        print(f"Launching tensorboard...")
        tensorboard_writer, tensorboard_url = create_tensorboard(database.path)
        print(f"Tensoboard is launched and accessible at: {tensorboard_url}")
    # print and save objective info
    obj_info = [
        f"Task: {task}",
        f"Evolver: {evolver}",
        f"Database path: {database.path}",
        f"Population size: {population_size}",
        f"Hyper-parameters: {len(_task.hyper_parameters)} {list(_task.hyper_parameters.keys())}",
        f"Batch size: {batch_size}",
        f"Train steps: {train_steps}",
        f"Fitness steps: {fitness_steps}",
        f"End criterium - steps: {end_steps}",
        f"End criterium - score: {end_score}",
        f"End criterium - time (in minutes): {end_time}",
        f"Loss-metric: {_task.loss_metric}",
        f"Eval-metric: {_task.eval_metric}",
        f"Loss functions: {[loss.name for loss in _task.loss_functions.values()]}",
        f"History limit: {history}",
        f"Training set length: {len(_task.datasets.train)}",
        f"Evaluation set length: {len(_task.datasets.eval)}",
        f"Testing set length: {len(_task.datasets.test)}",
        f"Total dataset length: {len(_task.datasets.train) + len(_task.datasets.eval) + len(_task.datasets.test)}",
        f"Verbosity: {verbose}",
        f"Logging: {logging}",
        f"Devices: {devices}",
        f"Number of processes: {n_jobs}"]
    if any(device.startswith('cuda') for device in devices):
        obj_info.append(f"Number of GPUs: {torch.cuda.device_count()}")
    obj_info = "\n".join(obj_info)
    print("\n", obj_info, "\n")
    database.create_file(
        tag="info", file_name="information.txt").write_text(obj_info)
    # define controller
    print(f"Creating evolver...")
    EVOLVER = create_evolver(manager, evolver, population_size, end_steps)
    # create controller
    if evolver == 'pbt':
        print(f"Creating PBT controller...")
        controller = PBTController(
            manager=manager,
            population_size=population_size,
            hyper_parameters=_task.hyper_parameters,
            evolver=EVOLVER,
            loss_metric=_task.loss_metric,
            eval_metric=_task.eval_metric,
            loss_functions=_task.loss_functions,
            database=database,
            end_criteria={'steps': end_steps, 'time': end_time, 'score': end_score},
            model_class=_task.model_class,
            optimizer_class=_task.optimizer_class,
            datasets=_task.datasets,
            batch_size=batch_size,
            train_steps=train_steps,
            devices=devices,
            n_jobs=n_jobs,
            history_limit=history,
            tensorboard=tensorboard_writer,
            verbose=verbose,
            logging=logging)
    else:
        print(f"Creating DE controller...")
        controller = DEController(
            manager=manager,
            population_size=population_size,
            hyper_parameters=_task.hyper_parameters,
            evolver=EVOLVER,
            loss_metric=_task.loss_metric,
            eval_metric=_task.eval_metric,
            loss_functions=_task.loss_functions,
            database=database,
            end_criteria={'steps': end_steps, 'time': end_time, 'score': end_score},
            model_class=_task.model_class,
            optimizer_class=_task.optimizer_class,
            datasets=_task.datasets,
            batch_size=batch_size,
            train_steps=train_steps,
            fitness_steps=fitness_steps,
            devices=devices,
            n_jobs=n_jobs,
            history_limit=history,
            tensorboard=tensorboard_writer,
            verbose=verbose,
            logging=logging)
    # run controller
    print(f"Starting controller...")
    best = controller.start()
    # analyze results stored in database
    print("Analyzing population...")
    analyzer = Analyzer(database, verbose=True)
    tester = Evaluator(
        model_class=_task.model_class, test_data=_task.datasets.test,
        loss_functions=_task.loss_functions, batch_size=batch_size,
        loss_group='test')
    analyzer.test(
        evaluator=tester,
        save_directory=database.create_file("results", "top_members.txt"),
        device='cpu')
    print("Creating statistics...")
    analyzer.create_statistics(
        save_directory=database.create_folder("results/statistics"))
    print("Creating plot-files...")
    analyzer.create_loss_plot_files(
        save_directory=database.create_folder("results/plots"))
    analyzer.create_time_plot_files(
        save_directory=database.create_folder("results/plots"))
    analyzer.create_hyper_parameter_plot_files(
        save_directory=database.create_folder("results/plots"))
    print("Program completed! You can now exit if needed.")
