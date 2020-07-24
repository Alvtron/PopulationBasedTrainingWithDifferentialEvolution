import os
import time
import dill
import copy
from datetime import datetime, timedelta
from abc import abstractmethod
from typing import List, Sequence, Iterable, Callable, Generator
from functools import partial
from multiprocessing.managers import SyncManager

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from pbt.device import DeviceCallable
from pbt.worker_pool import WorkerPool
from pbt.member import Checkpoint, Generation
from pbt.utils.date import get_datetime_string
from pbt.hyperparameters import DiscreteHyperparameter, Hyperparameters, hyper_parameter_change_details
from pbt.nn import Trainer, Evaluator, Step, RandomFitnessApproximation
from pbt.evolution import ExploitAndExplore, DifferentialEvolveEngine
from pbt.database import Database
from pbt.dataset import Datasets
from pbt.garbage import GarbageCollector

class Controller(object):
    def __init__(
            self, population_size: int, hyper_parameters: Hyperparameters, loss_metric: str, eval_metric: str,
            loss_functions: dict, database: Database, end_criteria: dict = {'score': 100.0}, verbose: int = 1,
            logging: bool = True, history_limit: int = None, tensorboard: SummaryWriter = None):
        if not isinstance(population_size, int):
            raise TypeError(f"the 'population_size' specified was of wrong type {type(population_size)}, expected {int}.")
        if not isinstance(hyper_parameters, Hyperparameters):
            raise TypeError(f"the 'hyper_parameters' specified was of wrong type {type(hyper_parameters)}, expected {Hyperparameters}.")
        if not isinstance(loss_metric, str):
            raise TypeError(f"the 'loss_metric' specified was of wrong type {type(loss_metric)}, expected {str}.")
        if not isinstance(eval_metric, str):
            raise TypeError(f"the 'eval_metric' specified was of wrong type {type(eval_metric)}, expected {str}.")
        if not isinstance(loss_functions, dict):
            raise TypeError(f"the 'loss_functions' specified was of wrong type {type(loss_functions)}, expected {dict}.")
        if not isinstance(database, Database):
            raise TypeError(f"the 'database' specified was of wrong type {type(database)}, expected {Database}.")
        if not isinstance(end_criteria, dict):
            raise TypeError(f"the 'end_criteria' specified was of wrong type {type(end_criteria)}, expected {dict}.")
        if not isinstance(verbose, int):
            raise TypeError(f"the 'verbose' specified was of wrong type {type(verbose)}, expected {int}.")
        if not isinstance(logging, bool):
            raise TypeError(f"the 'logging' specified was of wrong type {type(logging)}, expected {bool}.")
        if history_limit is not None and not isinstance(history_limit, int):
            raise TypeError(f"the 'history_limit' specified was of wrong type {type(history_limit)}, expected {int}.")
        if tensorboard is not None and not isinstance(tensorboard, SummaryWriter):
            raise TypeError(f"the 'tensorboard' specified was of wrong type {type(tensorboard)}, expected {SummaryWriter}.")
        self.population_size = population_size
        self.database = database
        self.hyper_parameters = hyper_parameters
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.loss_functions = loss_functions
        self.end_criteria = end_criteria
        self.verbose = verbose
        self.logging = logging
        self.__garbage_collector = GarbageCollector(
            database=database, history_limit=history_limit if history_limit and history_limit > 2 else 2, verbose=verbose > 2)
        self.__n_generations = 0
        self.__start_time = None
        self.__tensorboard = tensorboard

    @property
    def end_time(self):
        if self.__start_time is None or 'time' not in self.end_criteria or not self.end_criteria['time']:
            return None
        return self.__start_time + timedelta(minutes=self.end_criteria['time'])

    @abstractmethod
    def _print_prefix(self) -> str:
        raise NotImplementedError

    def __create_message(self, message: str) -> str:
        time = get_datetime_string()
        generation = f"G{self.__n_generations:03d}"
        prefixes = ' '.join(prefix for prefix in (
            time, generation, self._print_prefix()) if prefix is not None)
        return f"{prefixes}: {message}"

    def _say(self, message: str) -> None:
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 0."""
        self.__register(message=message, verbosity=0)

    def _whisper(self, message: str) -> None:
        """Prints the provided controller message in the appropriate syntax if verbosity level is above 1."""
        self.__register(message=message, verbosity=1)

    def __register(self, message: str, verbosity: int) -> None:
        """Logs and prints the provided message in the appropriate syntax. If a member is provided, the message is attached to that member."""
        file_tag = self.__class__.__name__
        full_message = self.__create_message(message)
        self.__print(message=full_message, verbosity=verbosity)
        self.__log_to_file(message=full_message, tag=file_tag)
        self.__log_to_tensorboard(
            message=full_message, tag=file_tag, global_steps=self.__n_generations)

    def __print(self, message: str, verbosity: int) -> None:
        if self.verbose > verbosity:
            print(message)

    def __log_to_file(self, message: str, tag: str) -> None:
        if not self.logging:
            return
        with self.database.create_file(tag='logs', file_name=f"{tag}_log.txt").open('a+') as file:
            file.write(message + '\n')

    def __log_to_tensorboard(self, message: str, tag: str, global_steps: int = None) -> None:
        if not self.__tensorboard:
            return
        self.__tensorboard.add_text(
            tag=tag, text_string=message, global_step=global_steps)

    def __update_tensorboard(self, member: Checkpoint) -> None:
        """Plots member data to tensorboard."""
        if self.__tensorboard is None:
            return
        # plot eval metrics
        for eval_metric_group, eval_metrics in member.loss.items():
            for metric_name, metric_value in eval_metrics.items():
                self.__tensorboard.add_scalar(
                    tag=f"metrics/{eval_metric_group}_{metric_name}/{member.uid:03d}",
                    scalar_value=metric_value,
                    global_step=member.steps)
        # plot time
        for time_type, time_value in member.time.items():
            self.__tensorboard.add_scalar(
                tag=f"time/{time_type}/{member.uid:03d}",
                scalar_value=time_value,
                global_step=member.steps)
        # plot hyper-parameters
        for hparam_name, hparam in member.parameters.items():
            self.__tensorboard.add_scalar(
                tag=f"hyperparameters/{hparam_name}/{member.uid:03d}",
                scalar_value=hparam.normalized if isinstance(
                    hparam, DiscreteHyperparameter) else hparam.value,
                global_step=member.steps)

    def __create_member(self, uid) -> Checkpoint:
        """Create a member object."""
        # create new member object
        member = Checkpoint(
            uid=uid,
            parameters=copy.deepcopy(self.hyper_parameters),
            loss_metric=self.loss_metric,
            eval_metric=self.eval_metric,
            minimize=self.loss_functions[self.eval_metric].minimize)
        return member

    def __create_members(self, k: int) -> List[Checkpoint]:
        members = list()
        for uid in range(k):
            members.append(self.__create_member(uid))
        return members

    def __create_initial_generation(self) -> Generation:
        new_members = self.__create_members(k=self.population_size)
        return new_members

    def __update_database(self, member: Checkpoint) -> None:
        """Updates the database stored in files."""
        self.database.update(member.uid, member.steps, member)

    def __is_score_end(self, generation):
        return 'score' in self.end_criteria and self.end_criteria['score'] and any(member >= self.end_criteria['score'] for member in generation)

    def __is_time_end(self):
        return 'time' in self.end_criteria and self.end_criteria['time'] and datetime.now() >= self.end_time

    def __is_n_generations_end(self):
        return 'generations' in self.end_criteria and self.end_criteria['generations'] and self.__n_generations >= self.end_criteria['generations']

    def _is_finished(self, generation: Generation) -> bool:
        """With the end_criteria, check if the generation is finished by inspecting the provided member."""
        if self.__is_score_end(generation):
            self._say("Score criterium has been met!")
            return True
        if self.__is_time_end():
            self._say("Time criterium has been met!")
            return True
        if self.__is_n_generations_end():
            self._say("Number of generations criterium has been met!")
            return True
        return False

    def start(self) -> Checkpoint:
        """Start global training procedure. Ends when end_criteria is met."""
        try:
            self._on_start()
            # start controller loop
            self._say("Starting training procedure...")
            return self.__train_synchronously()
        except KeyboardInterrupt:
            self._say("interupted.")
        except Exception:
            self._say("Exception not handled!")
            raise
        finally:
            self._say("finished.")
            self._on_stop()

    def _on_start(self) -> None:
        """Resets class properties, starts training service and cleans up temporary files."""
        # reset class properties
        self.__start_time = datetime.now()
        self.__n_generations = 0

    @abstractmethod
    def _on_stop(self) -> None:
        pass

    def __train_synchronously(self) -> Generation:
        """
        Performs the training of the population synchronously.
        Each member is trained individually and asynchronously,
        but they are waiting for each other between each generation cycle.
        """
        self._say("Creating initial generation...")
        initial_members = self.__create_initial_generation()
        for generation in self._train(initial_members):
            # Save member to database directory.
            self._whisper(f"saving members to database...")
            [self.__update_database(member) for member in generation]
            # write to tensorboard if enabled
            [self.__update_tensorboard(member) for member in generation]
            # perform garbage collection
            self._whisper("performing garbage collection...")
            self.__garbage_collector.collect(exclude=list(generation))
            # increment number of generations
            self.__n_generations += 1
        self._say(f"end criteria has been reached.")
        return max(generation)

    @abstractmethod
    def _train(self, initial: Iterable[Checkpoint]) -> Generator[List[Checkpoint], None, None]:
        raise NotImplementedError()

def is_ready(member: Checkpoint) -> bool:
    return True

class PBTController(Controller):
    def __init__(
            self, evolver: ExploitAndExplore, model_class, optimizer_class,
            datasets: Datasets, batch_size: int, train_steps: int, 
            manager: SyncManager, devices: Sequence[str] = ['cpu'], n_jobs: int = 1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(evolver, ExploitAndExplore):
            raise TypeError(f"the 'evolver' specified was of wrong type {type(evolver)}, expected {ExploitAndExplore}.")
        if not callable(model_class):
            raise TypeError(f"the 'model_class' specified was not callable.")
        if not callable(optimizer_class):
            raise TypeError(f"the 'optimizer_class' specified was not callable.")
        if not isinstance(datasets, Datasets):
            raise TypeError(f"the 'datasets' specified was of wrong type {type(datasets)}, expected {Datasets}.")
        if not isinstance(batch_size, int):
            raise TypeError(f"the 'batch_size' specified was of wrong type {type(batch_size)}, expected {int}.")
        if not isinstance(train_steps, int):
            raise TypeError(f"the 'train_steps' specified was of wrong type {type(train_steps)}, expected {int}.")
        if not isinstance(manager, SyncManager):
            raise TypeError(f"the 'manager' specified was of wrong type {type(manager)}, expected {SyncManager}.")
        if not isinstance(devices, (list, tuple)):
            raise TypeError(f"the 'devices' specified was of wrong type {type(devices)}, expected {list} or {tuple}.")
        if not isinstance(n_jobs, int):
            raise TypeError(f"the 'n_jobs' specified was of wrong type {type(n_jobs)}, expected {int}.")
        self._manager = manager
        self._worker_pool = WorkerPool(
            manager=manager, devices=devices, n_jobs=n_jobs, verbose=self.verbose - 3)
        # evolver
        self.evolver = evolver
        self.evolver.verbose = self.verbose > 2
        # create training function
        self.step_function = Step(
            model_class=model_class, optimizer_class=optimizer_class, train_data=datasets.train, test_data=datasets.eval,
            loss_functions=self.loss_functions, loss_metric=self.loss_metric, batch_size=batch_size, step_size=train_steps)
        # creating test function if test set is provided
        self.test_function = Evaluator(
            model_class=model_class, test_data=datasets.test, loss_functions=self.loss_functions,
            batch_size=batch_size, loss_group='test') if datasets.test else None

    def _print_prefix(self) -> str:
        if self.end_criteria['steps']:
            return f"({self.__n_steps}/{self.end_criteria['steps']})"
        else:
            return ""

    def _on_start(self) -> None:
        """Resets class properties, starts training service and cleans up temporary files."""
        super()._on_start()
        self.__n_steps = 0
        self._worker_pool.start()

    def _on_stop(self) -> None:
        """Stops training service and cleans up temporary files."""
        super()._on_stop()
        self._worker_pool.stop()

    def __is_steps_end(self):
        return 'steps' in self.end_criteria and self.end_criteria['steps'] and self.__n_steps >= self.end_criteria['steps']

    def _is_finished(self, generation: Generation):
        if super()._is_finished(generation):
            return True
        if self.__is_steps_end():
            self._say("Step criterium has been met!")
            return True
        return False

    def _train(self, initial: Iterable[Checkpoint]) -> Generator[List[Checkpoint], None, None]:
        # spawn members
        spawned_members = self.evolver.spawn(initial)
        # create generation
        generation = Generation(dict_constructor=dict, members=spawned_members)
        # create asynchronous procedure
        async_procedure = self.AsyncProcedure(
            generation=generation, train_function=self.step_function, evolver=self.evolver, test_function=self.test_function, verbose=self.verbose > 3)
        # loop until finished
        while not self._is_finished(generation):
            self._say("processing next generation...")
            for member in self._worker_pool.imap(async_procedure, list(generation), True):
                self.__n_steps += 1
                # report member performance
                self._say(f"{member}, {member.performance_details()}")
                self._whisper(f"{member}, {hyper_parameter_change_details(old_hps=generation[member.uid].parameters, new_hps=member.parameters)}")
                # update generation
                generation.update(member)
            yield list(generation)

    class AsyncProcedure(DeviceCallable):
        def __init__(self, generation: Generation, train_function, evolver: ExploitAndExplore, test_function=None, verbose: bool = False):
            super().__init__(verbose)
            if not isinstance(generation, Generation):
                raise TypeError(f"the 'generation' specified was of wrong type {type(generation)}, expected {Generation}.")
            if not callable(train_function):
                raise TypeError(f"the 'train_function' specified was not callable.")
            if not isinstance(evolver, ExploitAndExplore):
                raise TypeError(f"the 'evolver' specified was of wrong type {type(evolver)}, expected {ExploitAndExplore}.")
            if test_function is not None and not callable(test_function):
                raise TypeError(f"the 'test_function' specified was not callable.")
            self.generation = generation
            self.train_function = train_function
            self.evolver = evolver
            self.test_function = test_function

        def function(self, device: str, member: Checkpoint) -> Checkpoint:
            if not isinstance(member, Checkpoint):
                raise TypeError(f"the 'member' specified was of wrong type {type(member)}, expected {Checkpoint}.")
            if not isinstance(device, str):
                raise TypeError(f"the 'device' specified was of wrong type {type(device)}, expected {str}.")
            # train and evaluate
            self._print(f"training member {member.uid}...")
            train_start_time = datetime.now()
            self.train_function(checkpoint=member, device=device)
            member.register_time(tag='training', start=train_start_time, end=datetime.now())
            # exploit and explore
            self._print(f"mutating member {member.uid}...")
            evolve_start_time = datetime.now()
            member = self.evolver.mutate(member=member, generation=self.generation)
            member.register_time(tag='evolving', start=evolve_start_time, end=datetime.now())
            # measure test set performance if available
            if self.test_function is not None:
                test_start_time = datetime.now()
                self._print(f"testing member {member.uid}...")
                self.test_function(member, device)
                member.register_time(tag='testing', start=test_start_time, end=datetime.now())
            return member


class DEController(Controller):
    def __init__(
            self, evolver: DifferentialEvolveEngine, model_class, optimizer_class,
            datasets: Datasets, batch_size: int, train_steps: int, fitness_steps: int,
            manager: SyncManager, devices: Sequence[str] = ['cpu'], n_jobs: int = 1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(evolver, DifferentialEvolveEngine):
            raise TypeError(f"the 'evolver' specified was of wrong type {type(evolver)}, expected {DifferentialEvolveEngine}.")
        if not callable(model_class):
            raise TypeError(f"the 'model_class' specified was not callable.")
        if not callable(optimizer_class):
            raise TypeError(f"the 'optimizer_class' specified was not callable.")
        if not isinstance(datasets, Datasets):
            raise TypeError(f"the 'datasets' specified was of wrong type {type(datasets)}, expected {Datasets}.")
        if not isinstance(batch_size, int):
            raise TypeError(f"the 'batch_size' specified was of wrong type {type(batch_size)}, expected {int}.")
        if not isinstance(train_steps, int):
            raise TypeError(f"the 'train_steps' specified was of wrong type {type(train_steps)}, expected {int}.")
        if not isinstance(fitness_steps, int):
            raise TypeError(f"the 'fitness_steps' specified was of wrong type {type(fitness_steps)}, expected {int}.")
        if not isinstance(manager, SyncManager):
            raise TypeError(f"the 'manager' specified was of wrong type {type(manager)}, expected {SyncManager}.")
        if not isinstance(devices, (list, tuple)):
            raise TypeError(f"the 'devices' specified was of wrong type {type(devices)}, expected {list} or {tuple}.")
        if not isinstance(n_jobs, int):
            raise TypeError(f"the 'n_jobs' specified was of wrong type {type(n_jobs)}, expected {int}.")
        self._manager = manager
        self._worker_pool = WorkerPool(
            manager=manager, devices=devices, n_jobs=n_jobs, verbose=self.verbose - 3)
        # evolver
        self.evolver = evolver
        self.evolver.verbose = self.verbose > 2
        # create training function
        self.step_function = Step(
            model_class=model_class, optimizer_class=optimizer_class, train_data=datasets.train, test_data=datasets.eval,
            loss_functions=self.loss_functions, loss_metric=self.loss_metric, batch_size=batch_size, step_size=train_steps)
        # creating fitness function
        self.partial_fitness_function = partial(RandomFitnessApproximation,
            model_class=model_class, optimizer_class=optimizer_class, train_data=datasets.train, test_data=datasets.eval,
            loss_functions=self.loss_functions, loss_metric=self.loss_metric, batch_size=batch_size, batches=fitness_steps)
        # creating test function if test set is provided
        self.test_function = Evaluator(
            model_class=model_class, test_data=datasets.test, loss_functions=self.loss_functions,
            batch_size=batch_size, loss_group='test') if datasets.test else None
        self.__n_steps = 0

    def _print_prefix(self) -> str:
        if self.end_criteria['steps']:
            return f"({self.__n_steps}/{self.end_criteria['steps']})"
        else:
            return ""

    def _on_start(self) -> None:
        """Resets class properties, starts training service and cleans up temporary files."""
        super()._on_start()
        self.__n_steps = 0
        self._worker_pool.start()

    def _on_stop(self) -> None:
        """Stops training service and cleans up temporary files."""
        super()._on_stop()
        self._worker_pool.stop()

    def __is_steps_end(self):
        return 'steps' in self.end_criteria and self.end_criteria['steps'] and self.__n_steps >= self.end_criteria['steps']

    def _is_finished(self, generation: Generation):
        if super()._is_finished(generation):
            return True
        if self.__is_steps_end():
            self._say("Step criterium has been met!")
            return True
        return False

    def _train(self, initial: Iterable[Checkpoint]) -> Generator[List[Checkpoint], None, None]:
        # spawn members
        spawned_members = self.evolver.spawn(initial)
        # create generation
        generation = Generation(
            dict_constructor=self._manager.dict,
            members=spawned_members)
        # create asynchronous procedure
        async_procedure = self.AsyncProcedure(generation=generation, evolver=self.evolver, step_function=self.step_function,
            fitness_function=self.partial_fitness_function(), test_function=self.test_function, verbose=self.verbose > 3)
        while not self._is_finished(generation):
            # increment n steps
            self._whisper("on generation start...")
            self.evolver.on_generation_start(generation)
            for member in self._worker_pool.imap(async_procedure, list(generation), True):
                self.__n_steps += 1
                # report member performance
                self._say(f"{member}, {member.performance_details()}")
                self._whisper(f"{member}, {hyper_parameter_change_details(old_hps=generation[member.uid].parameters, new_hps=member.parameters)}")
                # update generation
                generation.update(member)
            self._whisper("on generation end...")
            self.evolver.on_generation_end(generation)
            yield list(generation)

    class AsyncProcedure(DeviceCallable):
        def __init__(
                self, generation: Generation, evolver: DifferentialEvolveEngine, step_function: Callable[[Checkpoint, str], None],
                fitness_function: Callable[[Checkpoint, str], None], test_function: Callable[[Checkpoint, str], None] = None, verbose: bool = False):
            super().__init__(verbose)
            if not isinstance(generation, Generation):
                raise TypeError(f"the 'generation' specified was of wrong type {type(generation)}, expected {Generation}.")
            if not isinstance(evolver, DifferentialEvolveEngine):
                raise TypeError(f"the 'evolver' specified was of wrong type {type(evolver)}, expected {DifferentialEvolveEngine}.")
            if not callable(step_function):
                raise TypeError(f"the 'step_function' specified was not callable.")
            if not callable(fitness_function):
                raise TypeError(f"the 'fitness_function' specified was not callable.")
            if test_function is not None and not callable(test_function):
                raise TypeError(f"the 'test_function' specified was not callable.")
            self.generation = generation
            self.evolver = evolver
            self.step_function = step_function
            self.fitness_function = fitness_function
            self.test_function = test_function

        def function(self, device: str, member: Checkpoint) -> Checkpoint:
            if not isinstance(member, Checkpoint):
                raise TypeError(f"the 'member' specified was of wrong type {type(member)}, expected {Checkpoint}.")
            if not isinstance(device, str):
                raise TypeError(f"the 'device' specified was of wrong type {type(device)}, expected {str}.")
            self._print(f"training member {member.uid}...")
            step_start_time = datetime.now()
            self.step_function(checkpoint=member, device=device)
            member.register_time(tag='training', start=step_start_time, end=datetime.now())
            self._print(f"evolving member {member.uid}...")
            evolve_start_time = datetime.now()
            member = self.evolver.mutate(
                parent=member,
                generation=self.generation,
                fitness_function=partial(self.fitness_function, device=device))
            member.register_time(tag='evolving', start=evolve_start_time, end=datetime.now())
            # measure test set performance if available
            if self.test_function is not None:
                test_start_time = datetime.now()
                self._print(f"testing member {member.uid}...")
                self.test_function(checkpoint=member, device=device)
                member.register_time(tag='testing', start=test_start_time, end=datetime.now())
            return member