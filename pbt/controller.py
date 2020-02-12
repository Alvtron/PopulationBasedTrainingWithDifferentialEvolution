import os
import sys
import math
import time
import random
import copy
import pickle
import shutil
from typing import List, Dict
from functools import partial 
from collections import defaultdict
from multiprocessing.context import BaseContext

import torch
import torchvision
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process
import matplotlib.pyplot as plt

import pbt.member
from .member import Checkpoint, Population, Generation, clear_member_states
from .worker import Worker, WorkerTask
from .utils.date import get_datetime_string
from .hyperparameters import DiscreteHyperparameter, Hyperparameters
from .trainer import Trainer
from .evaluator import Evaluator
from .evolution import EvolveEngine
from .database import Database

STOP_FLAG = None

class Controller(object):
    def __init__(
            self, context : BaseContext, population_size : int, hyper_parameters : Hyperparameters,
            trainer : Trainer, evaluator : Evaluator, evolver : EvolveEngine,
            loss_metric : str, eval_metric : str, loss_functions : dict, database : Database,
            step_size = 1, end_criteria : dict = {'score': 100.0}, eval_steps=1,
            tensorboard_writer : SummaryWriter = None, detect_NaN : bool = False, history_limit : int = None,
            device : str = 'cpu', n_jobs : int = -1, verbose : int = 1, logging : bool = True):
        assert step_size and step_size > 0, f"Step size must be of type {int} and 1 or higher."
        self.population = Population(population_size)
        self.database = database
        self.evolver = evolver
        self.hyper_parameters = hyper_parameters
        self.trainer = trainer
        self.evaluator = evaluator
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.step_size = step_size
        self.end_criteria = end_criteria
        self.eval_steps = eval_steps
        self.detect_NaN = detect_NaN
        self.device = device
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.logging = logging
        self.__tensorboard_writer = tensorboard_writer
        self.context = context
        # create shared event for when to end training
        self.end_event = self.context.Event()
        # create queues for training and evolving
        self.train_queue = self.context.Queue()
        self.evolve_queue = self.context.Queue()
        self.__workers = []
        self.__n_active_workers = 0
        self.history_limit = history_limit if history_limit and history_limit > 2 else 2
        # TODO: implement save and load function for controller
        self.save_objective_info()
        self.nfe = 0

    def save_objective_info(self):
        parameters = {
            'population_size': self.population.size,
            'batch_size': self.trainer.batch_size,
            'device': self.device,
            'hyper_parameters': self.hyper_parameters.keys(),
            'loss_metric': self.loss_metric,
            'eval_metric': self.eval_metric,
            'loss_functions': self.loss_functions,
            'step_size': self.step_size,
            'end_criteria': self.end_criteria
        }
        pickle.dump(parameters, self.database.create_file("info", "parameters.obj").open("wb"))

    def __str__(self):
        controller = self.__class__.__name__
        time = get_datetime_string()
        generation = f"G{len(self.population.generations):03d}"
        nfe = f"({self.nfe}/{self.end_criteria['nfe']})" if 'nfe' in self.end_criteria and self.end_criteria['nfe'] else self.nfe
        return f"{time} {nfe} {generation} {controller}"

    def __print(self, message : str):
        """Prints the provided controller message in the appropriate syntax."""
        print(f"{self}: {message}")

    def __say(self, message : str):
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 0."""
        if self.verbose > 0: self.__print(message)

    def __whisper(self, message : str):
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 1."""
        if self.verbose > 1: self.__print(message)

    def __log(self, member : Checkpoint, message : str, verbose=True):
        """Logs and prints the provided member message in the appropriate syntax."""
        time = get_datetime_string()
        nfe = f"({self.nfe}/{self.end_criteria['nfe']})" if 'nfe' in self.end_criteria and self.end_criteria['nfe'] else self.nfe
        full_message = f"{time} {nfe} G{len(self.population.generations):03d} {member}: {message}"
        if self.logging:
            with self.database.create_file(tag='logs', file_name=f"{member.id:03d}_log.txt").open('a+') as file:
                file.write(full_message + '\n')
        if verbose and self.verbose > 0:
            print(full_message)
        if self.__tensorboard_writer:
            self.__tensorboard_writer.add_text(tag=f"member_{member.id:03d}", text_string=full_message, global_step=member.steps)

    def __log_silent(self, member : Checkpoint, message : str):
        self.__log(member, message, verbose=self.verbose > 1)

    def garbage_collection(self):
        self.remove_bad_member_states()
        self.delete_inactive_workers()

    def remove_bad_member_states(self):
        if self.history_limit is None:
            return
        if len(self.population.generations) < self.history_limit + 1:
            return
        for member in self.population.generations[-(self.history_limit + 1)]:
            if not member.has_state():
                # skipping the member as it has no state to delete
                continue
            self.__whisper(f"deleting the state from member {member.id} at step {member.steps} with score {member.score():.2f}...")
            member.delete_state()
            # updating database
            self.database.update(member.id, member.steps, member)
        # clean tmp states
        for state_folder in pbt.member.MEMBER_STATE_DIRECTORY.glob("*"):
            step = int(state_folder.name)
            if step < len(self.population.generations) * self.step_size - self.history_limit * self.step_size: 
                shutil.rmtree(state_folder)

    def delete_inactive_workers(self):
        """Delete all inactive workers."""
        stopped_workers_idx = set(index for index, worker in enumerate(self.__workers) if not worker.is_alive())
        for index in sorted(stopped_workers_idx, reverse=True):
            self.__whisper(f"removing worker {self.__workers[index].id}...")
            del self.__workers[index]

    def adjust_workers_size(self):
        delta_size = self.__n_active_workers - self.population.size
        # adjust down if number of active workers is higher than the population size
        if delta_size > 0:
            self.stop_workers(k=delta_size)
        # adjust up if number of active workers is lower than the population size
        # and if adjustment does not exceed n_jobs
        elif delta_size < 0:
            new_size = min(abs(delta_size) + self.__n_active_workers, self.n_jobs)
            self.spawn_workers(k=new_size - self.__n_active_workers)

    def stop_workers(self, k : int = None):
        if not k:
            k = len(self.__workers)
        elif isinstance(k, int) and 0 < k < len(self.__workers):
            k = k
        else:
            raise IndexError()
        for i in range(k):
            self.train_queue.put(STOP_FLAG)
        self.__whisper(f"stopping {k} workers.")
        self.__n_active_workers -= k

    def create_workers(self, k):
        return [
            Worker(
                id=id,
                end_event_global=self.end_event,
                evolve_queue=self.evolve_queue,
                train_queue=self.train_queue,
                random_seed=id,
                verbose=self.verbose > 2)
            for id in range(k)]

    def __update_tensorboard(self):
        """Plots member data to tensorboard"""
        for member in self.population:
            # plot eval metrics
            for eval_metric_group, eval_metrics in member.loss.items():
                for metric_name, metric_value in eval_metrics.items():
                    self.__tensorboard_writer.add_scalar(
                        tag=f"metrics/{eval_metric_group}_{metric_name}/{member.id:03d}",
                        scalar_value=metric_value,
                        global_step=member.steps)
            # plot time
            for time_type, time_value in member.time.items():
                self.__tensorboard_writer.add_scalar(
                    tag=f"time/{time_type}/{member.id:03d}",
                    scalar_value=time_value,
                    global_step=member.steps)
            # plot hyper-parameters
            for hparam_name, hparam in member.hyper_parameters:
                self.__tensorboard_writer.add_scalar(
                    tag=f"hyperparameters/{hparam_name}/{member.id:03d}",
                    scalar_value=hparam.normalized if isinstance(hparam, DiscreteHyperparameter) else hparam.value,
                    global_step=member.steps)

    def is_member_ready(self, member : Checkpoint):
        """True every step size exceeded."""
        return member.steps % self.step_size == 0

    def is_member_finished(self, member : Checkpoint):
        """With the end_criteria, check if the provided member is finished training."""
        if 'steps' in self.end_criteria and self.end_criteria['steps'] and member.steps >= self.end_criteria['steps']:
            # the number of steps is equal or above the given treshold
            return True
        return False
    
    def is_population_finished(self):
        """
        With the end_criteria, check if the entire population is finished
        by inspecting the provided member.
        """
        if 'nfe' in self.end_criteria and self.end_criteria['nfe'] and self.nfe >= self.end_criteria['nfe']:
            return True
        if 'score' in self.end_criteria and self.end_criteria['score'] and any(not self.has_NaN_value(member) and member >= self.end_criteria['score'] for member in self.population):
            return True
        if all(self.is_member_finished(member) for member in self.population):
            return True
        return False

    def spawn_workers(self, k):
        """Spawn a worker process for every member in the population."""
        new_workers = self.create_workers(k)
        # Starting workers
        for index, worker in enumerate(new_workers, start=1):
            self.__whisper(f"starting worker {index}/{len(new_workers)}")
            worker.start()
        self.__n_active_workers += k
        self.__workers += new_workers

    def create_member(self, id):
        """Create a member object"""
        # create new member object
        member = Checkpoint(
            id=id,
            hyper_parameters=copy.deepcopy(self.hyper_parameters),
            loss_metric=self.loss_metric,
            eval_metric=self.eval_metric,
            minimize=self.loss_functions[self.eval_metric].minimize)
        # let evolver process new member
        self.evolver.on_member_spawn(member, self.__whisper)
        return member

    def create_members(self, k : int):
        members = list()
        for id in range(k):
            members.append(self.create_member(id))
        return members

    def queue_members(self, members : List[Checkpoint]):
        """Queue members for training."""
        for index, member in enumerate(members, start=1):
            self.train_member(member)
            self.__whisper(f"queued member {index}/{len(members)} for training...")

    def has_NaN_value(self, member : Checkpoint):
        """Returns True if the member has NaN values for its loss_metric or eval_metric."""
        for loss_type, loss_value in member.loss['eval'].items():
            if loss_type in (self.loss_metric, self.eval_metric) and math.isnan(loss_value):
                return True
        return False

    def on_NaN_detect(self, member : Checkpoint):
        """
        Handle member with NaN loss. Depending on the active population size,
        generate a new member start-object or sample one from the previous generation.
        """
        # create new member if population is not fully initialized yet
        if len(self.population.generations) < 2:
            self.replace_member_with_new(member)
        else:
            self.replace_member_with_best(member)

    def replace_member_with_new(self, member : Checkpoint):
        """Replace member with a new member."""
        self.__log_silent(member, "creating new member...")
        new_member = self.create_member(member.id)
        member.delete_state()
        member.update(new_member)
        member.steps = new_member.steps
        member.epochs = new_member.epochs

    def replace_member_with_best(self, member : Checkpoint, p = 0.2):
        """Replace member with one of the best performing members in a previous generation."""
        previous_generation = self.population.generations[-2]
        n_elitists = max(1, round(len(previous_generation) * p))
        elitists = sorted((m for m in previous_generation if m != member), reverse=True)[:n_elitists]
        elitist = random.choice(elitists)
        self.__log_silent(member, f"replacing with member {elitist.id}...")
        member.update(elitist)
        # resample hyper parameters
        [hp.sample_uniform() for hp in member]

    def eval_function(self, candidate : Checkpoint):
        """
        Evaluates the provided candidate checkpoint.
        Returns the candidate checkpoint.
        The checkpoint object holds the loss result.
        """
        # load candidate state
        candidate.load_state()
        # train
        candidate.model_state, candidate.optimizer_state, candidate.epochs, candidate.steps, candidate.loss['train'] = self.trainer.train(
            hyper_parameters=candidate.hyper_parameters,
            model_state=candidate.model_state,
            optimizer_state=candidate.optimizer_state,
            epochs=candidate.epochs,
            steps=candidate.steps,
            step_size=self.eval_steps
            )
        # eval
        candidate.loss['eval'] = self.evaluator.eval(candidate.model_state)
        # unload candidate state
        candidate.unload_state()
        return candidate

    def start_training_procedure(self):
        """
        Creates worker-processes for training and evaluation.
        Creates, prepares and queue member-objects for training.
        """
        self.__whisper("creating initial members member objects...")
        members = self.create_members(self.population.size)
        self.__whisper("creating worker processes...")
        n_workers = self.n_jobs if 0 < self.n_jobs <= self.population.size else self.population.size
        self.spawn_workers(n_workers)
        self.__whisper("queuing member members...")
        self.queue_members(members)

    def start(self):
        """
        Start global training procedure. Ends when end_criteria is met.
        """
        # reset member state folder
        clear_member_states()
        try:
            # start controller loop
            self.__say("Starting training procedure...")
            self.train_synchronously()
            # terminate worker processes
            self.__say("controller is finished.")
        except KeyboardInterrupt:
            self.__say("controller was interupted.")
        self.__whisper("sending stop signal to the remaining worker processes...")
        self.stop_workers()
        self.__whisper("waiting for all worker processes to finish...")
        [worker.join() for worker in self.__workers]
        self.delete_inactive_workers()
        self.__whisper("termination was successfull!")
        # reset member state folder
        clear_member_states()
        # clear CUDA memory
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

    def train_synchronously(self):
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self.start_training_procedure()
        while not self.end_event.is_set():
            while not self.population.full():
                # get next member
                self.__whisper("awaiting next trained member...")
                member = self.evolve_queue.get()
                # if a STOP FLAG is received, stop the training process
                if member is STOP_FLAG:
                    return
                # if the member is ahead of any remaining members from the previous generation,
                # place it in the back of the queue and wait a little
                if member.steps > len(self.population.generations) * self.step_size + self.step_size:
                    self.__log_silent(member, "ahead of the previous generation: Waiting...")
                    time.sleep(0.5)
                    self.evolve_queue.put(member)
                    continue
                # check for nan loss if enabled
                if self.detect_NaN and self.has_NaN_value(member):
                    self.__log_silent(member, "NaN metric detected.")
                    self.on_NaN_detect(member)
                    self.train_member(member)
                    continue
                # log performance
                self.__log(member, member.performance_details())
                # Add member to population.
                self.__log_silent(member, "updating population...")
                self.population.append(member)
                # Save member to database directory.
                self.__log_silent(member, "updating database...")
                member.load_state()
                self.database.update(member.id, member.steps, member)
                member.unload_state()
            # write to tensorboard if enabled
            if self.__tensorboard_writer:
                self.__update_tensorboard()
            # check if population is finished
            if self.is_population_finished():
                self.__say(f"end criteria has been reached.")
                self.end_event.set()
                return
            # Start evolving
            self.__whisper("on generation start")
            self.evolver.on_generation_start(self.population, self.__whisper)
            for member in self.population:
                # make a copy of the member
                candidate = member.copy()
                # evolve member if ready
                if self.is_member_ready(member):
                    self.evolve_member(candidate)
                # train candidate
                self.train_member(candidate)
            self.__whisper("on generation end")
            self.evolver.on_generation_end(self.population, self.__whisper)
            # perform garbage collection and adjust worker size
            self.__whisper("performing garbage collection...")
            self.garbage_collection()
            self.__whisper("adjusting the number of workers...")
            self.adjust_workers_size()
            # create new generation
            self.population.new_generation()

    def evolve_member(self, member : Checkpoint):
        """Evolve the provided member checkpoint with the evolve engine."""
        candidate = member.copy()
        self.__log_silent(member, "evolving...")
        start_evolve_time_ns = time.time_ns()
        candidate = self.evolver.on_evolve(
            member=candidate,
            generation=self.population,
            logger=partial(self.__log_silent, member))
        self.__log_silent(member, "evaluating mutant...")
        candidate = self.evolver.on_evaluation(
            member=member,
            candidate=candidate,
            eval_function=self.eval_function,
            logger=partial(self.__log_silent, member))
        self.nfe += 1
        self.__log_silent(member, "updating...")
        member.update(candidate)
        member.time['evolve'] = float(time.time_ns() - start_evolve_time_ns) * float(10**(-9))

    def train_member(self, member : Checkpoint):
        """Queue the provided member checkpoint for training."""
        # adjust step size
        step_size = self.step_size - (member.steps % self.step_size)
        # create worker task
        task = WorkerTask(member, self.trainer, self.evaluator, step_size=step_size)
        self.__log_silent(member, "training...")
        self.train_queue.put(task)