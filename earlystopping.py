class EarlyStopping(object):
    def __init__(self, minimize, a, k = 5, s = 5, criteria = 'PQ'):
        """
        a (float): generalization loss threshold
        k (int): training strip length
        s (int): error increase in s successive strips

        Implementation of methods for overfitting detection from
        "Early Stopping - but when?" (1997) by Lutz Prechelt (prechelt@ira.uka.de)
        """
        self.minimize = minimize
        self.train_scores = list()
        self.eval_scores = list()
        self.criteria = criteria
        self.a = a
        self.k = k
        self.s = s
        self.s_count = 0
    
    @property
    def current(self) -> float:
        return self.eval_scores[-1] if self.eval_scores else None
    
    @property
    def best(self) -> float:
        """
        The best validation set error obtained so far
        """
        if not self.eval_scores:
            return None
        return min(self.eval_scores) if self.minimize else max(self.eval_scores)

    def generalization_loss(self) -> float:
        """
        The generalization loss (in percent) at interval t.
        It is the relative increase of the validation error over the best-so-far.
        
        High generalization loss is one obvious candidate reason to stop training, be-
        cause it directly indicates overtting.
        """
        loss = self.current / self.best if self.minimize else self.best / self.current
        return 100.0 * (loss - 1)

    def training_progress_loss(self):
        a = sum(self.train_scores[-self.k])
        b = self.k * min(self.train_scores[-self.k]) if self.minimize else max(self.train_scores[-self.k])
        return 1000 * ((a / b) - 1)

    def generalization_loss_quotient(self):
        """
        The quotient of generalization loss and progress
        """
        return self.generalization_loss() / self.training_progress_loss()

    def register(self, train_score, eval_score):
        self.train_scores.append(train_score)
        self.eval_scores.append(eval_score) 

    def first_criteria(self):
        """
        GL: True when the generalization loss exceeds threshold 'a'.
        """
        return self.generalization_loss() > self.a

    def second_criteria(self):
        """
        PQ: True when use the quotient of generalization loss and progress is higher than threshold 'a'.
        """
        if len(self.eval_scores) < 5:
            return False
        return self.generalization_loss_quotient() > self.a

    def third_criteria(self):
        """
        UP: True when the generalization error increased in 's' successive strips.

        When the validation error has increased not only once, but during 's' consecutive strips,
        we assume that such increases indicate the beginning of nal overtting
        """
        if len(self.eval_scores) < 5:
            return False
        last_k_scores = self.eval_scores[-self.k]
        a = self.minimize and all(i < j for i, j in zip(last_k_scores, last_k_scores[1:]))
        b = not self.minimize and all(i > j for i, j in zip(last_k_scores, last_k_scores[1:]))
        if a or b:
            self.s_count += 1
        return self.s_count == self.s

    def is_criteria_met(self) -> bool:
        a = self.criteria == 'GL' and self.first_criteria()
        b = self.criteria == 'PQ' and self.second_criteria()
        c = self.criteria == 'UP' and self.third_criteria()
        return a or b or c