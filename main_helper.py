import os
import copy
import random
import time
import argparse
import multiprocessing
from typing import List
from functools import partial
from dataclasses import dataclass

import torch
import numpy as np
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import pbt.evolution
from pbt.controller import Controller
from pbt.task import mnist, fashionmnist, creditfraud
from pbt.analyze import Analyzer
from pbt.database import Database
from pbt.evaluator import Evaluator
from pbt.trainer import Trainer

def import_task(task_name : str):
    if task_name == "creditfraud":
        return creditfraud.CreditCardFraud()
    elif task_name == "mnist":
        return mnist.Mnist()
    elif task_name == "mnist_lenet5":
        return mnist.Mnist('lenet5')
    elif task_name == "mnist_mlp":
        return mnist.Mnist('mlp')
    elif task_name == "mnist_knowledge_sharing_lenet5":
        return mnist.MnistKnowledgeSharing('lenet5')
    elif task_name == "mnist_knowledge_sharing_mlp":
        return mnist.MnistKnowledgeSharing('mlp')
    elif task_name == "emnist_byclass":
        return mnist.EMnist("byclass")
    elif task_name == "emnist_bymerge":
        return mnist.EMnist("bymerge")
    elif task_name == "emnist_balanced":
        return mnist.EMnist("balanced")
    elif task_name == "emnist_letters":
        return mnist.EMnist("letters")
    elif task_name == "emnist_digits":
        return mnist.EMnist("digits")
    elif task_name == "emnist_mnist":
        return mnist.EMnist("mnist")
    elif task_name == "fashionmnist":
        return fashionmnist.FashionMnist('default')
    elif task_name == "fashionmnist_lenet5":
        return fashionmnist.FashionMnist('lenet5')
    elif task_name == "fashionmnist_mlp":
        return fashionmnist.FashionMnist('mlp')
    elif task_name == "fashionmnist_knowledge_sharing_lenet5":
        return fashionmnist.FashionMnistKnowledgeSharing('lenet5')
    elif task_name == "fashionmnist_knowledge_sharing_mlp":
        return fashionmnist.FashionMnistKnowledgeSharing('mlp')
    else:
        raise NotImplementedError(f"Your requested task '{task_name}'' is not available.")

def create_evolver(evolver_name, population_size, end_nfe):
    if evolver_name == 'rw':
        return pbt.evolution.RandomWalk(explore_factor = 0.2)
    if evolver_name == 'pbt':
        return pbt.evolution.ExploitAndExplore( exploit_factor = 0.2, explore_factors = (0.9, 1.1))
    if evolver_name == 'pbt_de':
        return pbt.evolution.ExploitAndExploreWithDifferentialEvolution( exploit_factor = 0.2, F = 0.2, Cr = 0.8)
    if evolver_name == 'de':
        return pbt.evolution.DifferentialEvolution(F = 0.2, Cr = 0.8)
    if evolver_name == 'shade':
        return pbt.evolution.SHADE(N_INIT = population_size, r_arc=2.0, p=0.2, memory_size=5)
    if evolver_name == 'lshade':
        return pbt.evolution.LSHADE(N_INIT = population_size, MAX_NFE=end_nfe, r_arc=2.0, p=0.2, memory_size=5)
    if evolver_name == 'decaying_lshade_linear':
        return pbt.evolution.DecayingLSHADE(N_INIT = population_size, MAX_NFE=end_nfe, r_arc=2.0, p=0.2, memory_size=5, type='linear')
    if evolver_name == 'decaying_lshade_curve':
        return pbt.evolution.DecayingLSHADE(N_INIT = population_size, MAX_NFE=end_nfe, r_arc=2.0, p=0.2, memory_size=5, type='curve')
    if evolver_name == 'decaying_lshade_logistic':
        return pbt.evolution.DecayingLSHADE(N_INIT = population_size, MAX_NFE=end_nfe, r_arc=2.0, p=0.2, memory_size=5, type='logistic')
    if evolver_name == 'lshadewithweightsharing':
        return pbt.evolution.LSHADEWithWeightSharing(N_INIT = population_size, MAX_NFE=end_nfe, r_arc=2.0, p=0.2, memory_size=5)
    else:
        raise NotImplementedError(f"Your evolver request '{evolver_name}'' is not available.")

def create_tensorboard(log_directory):
    tensorboard_log_path = f"{log_directory}/tensorboard"
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tensorboard_log_path])
    url = tb.launch()
    return SummaryWriter(tensorboard_log_path), url

def run(task : str, evolver : str, population_size : int, batch_size : int, step_size : int,
        end_nfe : int = None, end_steps : int = None, end_score : float = None, history : int = 2,
        directory : str = 'checkpoints', devices : List[str] = ['cpu'], n_jobs : int = 1,
        tensorboard : bool = False, detect_NaN : bool = False, old_controller : bool = False,
        verbose : int = 1, logging : bool = True):
    # prepare objective
    print(f"Importing task...")
    _task = import_task(task)
    # prepare database
    print(f"Preparing database...")
    database = Database(
        directory_path=f"{directory}/{'old' if old_controller else 'new'}_{task}_p{population_size}_steps{step_size}_batch{batch_size}_nfe{end_nfe}_{evolver}",
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
        f"Hyper-parameters: {len(_task.hyper_parameters)} {_task.hyper_parameters.keys()}",
        f"Batch size: {batch_size}",
        f"Step size: {step_size}",
        f"End criterium - fitness evaluations: {end_nfe}",        
        f"End criterium - steps: {end_steps}",
        f"End criterium - score: {end_score}",
        f"Loss-metric: {_task.loss_metric}",
        f"Eval-metric: {_task.eval_metric}",
        f"Loss functions: {[loss.name for loss in _task.loss_functions.values()]}",
        f"History limit: {history}",
        f"Detect NaN: {detect_NaN}",
        f"Training set length: {len(_task.datasets.train)}",
        f"Evaluation set length: {len(_task.datasets.eval)}",
        f"Testing set length: {len(_task.datasets.test)}",
        f"Total dataset length: {len(_task.datasets.train) + len(_task.datasets.eval) + len(_task.datasets.test)}",
        f"Verbosity: {verbose}",
        f"Logging: {logging}",
        f"Devices: {devices}",
        f"Number of processes: {n_jobs}"]
    #if devices == "cuda":
    #    obj_info.append(f"Number of GPUs: {torch.cuda.device_count()}")
    obj_info = "\n".join(obj_info)
    print("\n", obj_info, "\n")
    database.create_file(tag="info", file_name="information.txt").write_text(obj_info)
    # create trainer, evaluator and tester
    print(f"Creating trainer...")
    TRAINER = Trainer(
        model_class = _task.model_class,
        optimizer_class = _task.optimizer_class,
        train_data = _task.datasets.train,
        loss_functions = _task.loss_functions,
        loss_metric = _task.loss_metric,
        batch_size = batch_size,
        verbose=False)
    print(f"Creating evaluator...")
    EVALUATOR = Evaluator(
        model_class = _task.model_class,
        test_data = _task.datasets.eval,
        loss_functions=_task.loss_functions,
        batch_size = batch_size,
        loss_group = 'eval',
        verbose=False)
    print(f"Creating tester...")
    TESTER = Evaluator(
        model_class = _task.model_class,
        test_data = _task.datasets.test,
        loss_functions=_task.loss_functions,
        batch_size = batch_size,
        loss_group = 'test',
        verbose=False)
    # define controller
    print(f"Creating evolver...")
    EVOLVER = create_evolver(evolver, population_size, end_nfe)
    # create controller
    print(f"Creating controller...")
    controller = Controller(
        population_size=population_size,
        hyper_parameters=_task.hyper_parameters,
        trainer=TRAINER,
        evaluator=EVALUATOR,
        evolver=EVOLVER,
        loss_metric=_task.loss_metric,
        eval_metric=_task.eval_metric,
        loss_functions=_task.loss_functions,
        database=database,
        step_size=step_size,
        end_criteria={'nfe': end_nfe, 'steps': end_steps, 'score': end_score},
        detect_NaN=detect_NaN,
        devices=devices,
        n_jobs=n_jobs,
        history_limit=history,
        tensorboard_writer=tensorboard_writer,
        verbose=verbose,
        logging=logging)
    # run controller
    print(f"Starting controller...")
    controller.start(use_old = old_controller) 
    # analyze results stored in database
    print("Analyzing population...")
    analyzer = Analyzer(database, verbose=True)
    checkpoint = analyzer.test(
        evaluator=TESTER,
        save_directory=database.create_file("results", "top_members.txt"),
        device='cpu')
    print("Updating database with results...")
    database.update(checkpoint.id, checkpoint.steps, checkpoint)
    print("Creating statistics...")
    analyzer.create_statistics(save_directory=database.create_folder("results/statistics"))
    print("Creating plot-files...")
    analyzer.create_plot_files(save_directory=database.create_folder("results/plots"))
    analyzer.create_hyper_parameter_plot_files(save_directory=database.create_folder("results/plots"), sensitivity=20)
    print("Program completed! You can now exit if needed.")