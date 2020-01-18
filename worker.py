import torch
import time
import random
import numpy
import uuid

# set random state for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mp = torch.multiprocessing.get_context('spawn')

class Worker(mp.Process):
    """A worker process that train and evaluate any available checkpoints provided from the train_queue. """
    def __init__(self, id, end_event_global, end_event_private, evolve_queue, train_queue, trainer, evaluator, random_seed = 0):
        super().__init__()
        self.id = id
        self.end_event_global = end_event_global
        self.end_event_private = end_event_private
        self.evolve_queue = evolve_queue
        self.train_queue = train_queue
        self.trainer = trainer
        self.evaluator = evaluator
        # set random state for reproducibility
        random.seed(random_seed)
        numpy.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def run(self):
        print(f"Worker {self.id} is running...")
        while not self.end_event_global.is_set() and not self.end_event_private.is_set():
            # get next checkpoint from train queue
            checkpoint = self.train_queue.get()
            if checkpoint is None:
                del checkpoint
                break
            # train checkpoint model
            start_train_time_ns = time.time_ns()
            checkpoint.model_state, checkpoint.optimizer_state, checkpoint.epochs, checkpoint.steps, checkpoint.loss['train'] = self.trainer.train(
                hyper_parameters=checkpoint.hyper_parameters,
                model_state=checkpoint.model_state,
                optimizer_state=checkpoint.optimizer_state,
                epochs=checkpoint.epochs,
                steps=checkpoint.steps,
                step_size=checkpoint.step_size)
            checkpoint.time['train'] = float(time.time_ns() - start_train_time_ns) * float(10**(-9))
            # evaluate checkpoint model
            start_eval_time_ns = time.time_ns()
            checkpoint.loss['eval'] = self.evaluator.eval(checkpoint.model_state)
            checkpoint.time['eval'] = float(time.time_ns() - start_eval_time_ns) * float(10**(-9))
            self.evolve_queue.put(checkpoint)
            # release memory
            del checkpoint
        print(f"Worker {self.id} has stopped.")