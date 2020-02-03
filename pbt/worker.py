import os
import torch
import time
import random
import numpy

from .member import Checkpoint

STOP_FLAG = None

# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
# multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')
mp = torch.multiprocessing.get_context('spawn')

class Worker(mp.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """
    def __init__(self, id, end_event_global, evolve_queue, train_queue, trainer, evaluator, random_seed : int = 0, verbose : bool = False):
        super().__init__()
        self.id = id
        self.end_event_global = end_event_global
        self.evolve_queue = evolve_queue
        self.train_queue = train_queue
        self.trainer = trainer
        self.evaluator = evaluator
        self.verbose = verbose
        # set random state for reproducibility
        random.seed(random_seed)
        numpy.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def __log(self, message : str):
        prefix = f"Worker {self.id}, PID {os.getpid()}"
        full_message = f"{prefix}: {message}"
        if self.verbose:
            print(full_message)

    def run(self):
        self.__log("running...")
        while not self.end_event_global.is_set():
            self.__log("awaiting checkpoint...")
            # get next checkpoint from train queue
            checkpoint : Checkpoint = self.train_queue.get()
            if checkpoint is STOP_FLAG:
                del checkpoint
                continue
            try:
                # load state
                self.__log("loading checkpoint state...")
                model_state, optimizer_state = checkpoint.load_state()
                if (model_state is None or optimizer_state is None) and checkpoint.steps >= checkpoint.step_size:
                    self.__log(f"WARNING: received trained checkpoint {checkpoint.id} at step {checkpoint.steps} with missing state-files.")
                # train checkpoint model
                start_train_time_ns = time.time_ns()
                self.__log("training...")
                model_state, optimizer_state, checkpoint.epochs, checkpoint.steps, checkpoint.loss['train'] = self.trainer.train(
                    hyper_parameters=checkpoint.hyper_parameters,
                    model_state=model_state,
                    optimizer_state=optimizer_state,
                    epochs=checkpoint.epochs,
                    steps=checkpoint.steps,
                    step_size=checkpoint.step_size)
                checkpoint.time['train'] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))
                # save state
                self.__log("saving checkpoint state...")
                checkpoint.save_state(model_state, optimizer_state)
                # evaluate checkpoint model
                start_eval_time_ns = time.time_ns()
                self.__log("evaluating...")
                checkpoint.loss['eval'] = self.evaluator.eval(model_state)
                checkpoint.time['eval'] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))
                self.__log(f"Time: {checkpoint.time['train']:.2f}s train, {checkpoint.time['eval']:.2f}s eval")
            except Exception as e:
                self.__log(e)
                raise e
                #TODO: recollect checkpoint and queue it for training
            self.evolve_queue.put(checkpoint)
            self.__log("checkpoint returned.")
        self.__log("stopped.")