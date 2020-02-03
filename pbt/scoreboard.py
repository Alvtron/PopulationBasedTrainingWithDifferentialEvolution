import collections

from .loss import _Loss

class Scoreboard(object):
    def __init__(self, loss_metric, eval_metric, *args):
        assert len(args) > 1
        self.__groups = ["train", "eval", "test"]
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric
        self.functions = dict()
        for metric in args:
            assert isinstance(metric, _Loss)
            self.functions[metric.iso] = metric
        self.values = {x: dict.fromkeys(self.functions.keys(), 0.0) for x in self.__groups}

    @property
    def loss_value(self):
        return self.values["eval"][self.loss_metric]

    @property
    def eval_value(self):
        return self.values["eval"][self.eval_metric]

    @property
    def loss_function(self):
        return self.functions[self.loss_metric]

    @property
    def eval_function(self):
        return self.functions[self.eval_metric]

    def __str__(self):
        string = ""
        for group, values in self.values.items():
            for metric, value in values.items():
                string += f", {group}_{metric} {value:.5f}"
        return string

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            return self.eval_value > other if self.eval_function.minimize else self.eval_value < other
        else:
            assert other.eval_metric == self.eval_metric, "Eval metrics are not the same."
            return self.eval_value > other.eval_value if self.eval_function.minimize else self.eval_value < other.eval_value

    def __gt__(self, other):
        if isinstance(other, (float, int)):
            return self.eval_value < other if self.eval_function.minimize else self.eval_value > other
        else:
            assert other.eval_metric == self.eval_metric, "Eval metrics are not the same."
            return self.eval_value < other.eval_value if self.eval_function.minimize else self.eval_value > other.eval_value

    def __le__(self, other):
        if isinstance(other, (float, int)):
            return self.eval_value <= other if self.eval_function.minimize else self.eval_value >= other
        else:
            assert other.eval_metric == self.eval_metric, "Eval metrics are not the same."
            return self.eval_value <= other.eval_value if self.eval_function.minimize else self.eval_value >= other.eval_value

    def __ge__(self, other):
        if isinstance(other, (float, int)):
            return self.eval_value >= other if self.eval_function.minimize else self.eval_value <= other
        else:
            assert other.eval_metric == self.eval_metric, "Eval metrics are not the same."
            return self.eval_value >= other.eval_value if self.eval_function.minimize else self.eval_value <= other.eval_value

    def update(self, group, metric, value):
        self.values[group][metric] = value

    def reset(self, metric, group=None):
        if group:
            self.values[group] = dict.fromkeys(self.__groups, 0.0)
        else:
            self.values = {x: dict.fromkeys(self.functions.keys(), 0.0) for x in self.__groups}