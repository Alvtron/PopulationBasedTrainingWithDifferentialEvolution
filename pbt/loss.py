import math
from abc import abstractmethod

import torch
from sklearn.metrics import f1_score, precision_score, recall_score


class _Loss(torch.nn.Module):
    '''
    Base class for torch loss functions, with extra information.
    Requires specification of in which direction the score is considered better.
    '''
    def __init__(self, name: str, iso: str, minimum: float, maximum: float, minimize: bool):
        super().__init__()
        if not isinstance(name, str):
            raise TypeError(f"the 'name' specified was of wrong type {type(name)}, expected {str}.")
        if not isinstance(iso, str):
            raise TypeError(f"the 'iso' specified was of wrong type {type(iso)}, expected {str}.")
        if not isinstance(minimum, float):
            raise TypeError(f"the 'minimum' specified was of wrong type {type(minimum)}, expected {float}.")
        if not isinstance(maximum, float):
            raise TypeError(f"the 'maximum' specified was of wrong type {type(maximum)}, expected {float}.")
        self.name = name
        self.iso = iso
        self.minimum = minimum
        self.maximum = maximum
        self.minimize = minimize

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Accuracy(_Loss):
    '''
    Calculate the accuracy.
    '''
    def __init__(self, in_decimal: bool = False):
        super().__init__(name="Accuracy", iso="acc", minimum=0.0, maximum=1.0 if in_decimal else 100.0, minimize=False)
        self.in_decimal = in_decimal
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n_samples = torch.numel(y_true)
        predicted = torch.argmax(y_pred, dim=1)
        n_correct = predicted.eq(y_true).sum().float()
        accuracy = n_correct / float(n_samples)
        return accuracy if self.in_decimal else accuracy * 100.0


class MAE(_Loss):
    '''
    Calculate the mean absolute error (L1 norm).
    '''
    def __init__(self, **kwargs):
        super().__init__(name="Mean Absolute Error", iso="l1", minimum=0.0, maximum=math.inf, minimize=True)
        self.function = torch.nn.L1Loss(**kwargs)
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.function(y_pred, y_true)


class MSE(_Loss):
    '''
    Calculate the mean squared error (squared L2 norm).
    '''
    def __init__(self, **kwargs):
        super().__init__(name="Mean Squared Error", iso="l2", minimum=0.0, maximum=math.inf, minimize=True)
        self.function = torch.nn.MSELoss(**kwargs)
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.function(y_pred, y_true)


class CategoricalCrossEntropy(_Loss):
    '''
    Calculate the cross-entropy loss.
    '''
    def __init__(self, **kwargs):
        super().__init__(name="Categorical Cross Entropy", iso="cce", minimum=0.0, maximum=math.inf, minimize=True)
        self.function = torch.nn.CrossEntropyLoss(**kwargs)
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.function(y_pred, y_true)


class BinaryCrossEntropy(_Loss):
    '''
    Calculate the binary cross-entropy loss.
    '''
    def __init__(self, **kwargs):
        super().__init__(name="Categorical Cross Entropy", iso="cce", minimum=0.0, maximum=math.inf, minimize=True)
        self.function = torch.nn.BCELoss(**kwargs)
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.function(y_pred, y_true)


class NLL(_Loss):
    '''
    Calculate the negative log likelihood loss.
    '''
    def __init__(self, **kwargs):
        super().__init__(name="Negative Log Likelihood", iso="nll", minimum=0.0, maximum=math.inf, minimize=True)
        self.function = torch.nn.NLLLoss(**kwargs)
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.function(y_pred, y_true)    


class F1(_Loss):
    '''
    Calculate the macro F1 score.
    F1 score is also known as balanced F-score or F-measure.
    It is the weighted average of the precision and sensitivity (recall).
    '''
    def __init__(self, classes: int, epsilon: float = 1e-12):
        super().__init__(name="F1 Score", iso="f1", minimum=0.0, maximum=1.0, minimize=False)
        if not isinstance(classes, int):
            raise TypeError(f"the 'classes' specified was of wrong type {type(classes)}, expected {int}.")
        if not isinstance(epsilon, float):
            raise TypeError(f"the 'epsilon' specified was of wrong type {type(epsilon)}, expected {float}.")
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # convert from class probability to class labels
        y_pred = torch.argmax(y_pred, 1)
        return f1_score(y_true=y_true.cpu(), y_pred=y_pred.cpu(), labels=list(range(self.classes)), average='macro', zero_division=0)


class Sensitivity(_Loss):
    '''
    Calculate the sensitivity score.
    Sensitivity is also called the true positive rate, the recall, or probability of detection.
    It is the fraction of the total amount of relevant instances that were actually retrieved.
    '''
    def __init__(self, classes: int, epsilon: float = 1e-12):
        super().__init__(name="Sensitivity", iso="sens", minimum=0.0, maximum=1.0, minimize=False)
        if not isinstance(classes, int):
            raise TypeError(f"the 'classes' specified was of wrong type {type(classes)}, expected {int}.")
        if not isinstance(epsilon, float):
            raise TypeError(f"the 'epsilon' specified was of wrong type {type(epsilon)}, expected {float}.")
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # convert from class probability to class labels
        y_pred = torch.argmax(y_pred, 1)
        return recall_score(y_true=y_true.cpu(), y_pred=y_pred.cpu(), labels=list(range(self.classes)), average='macro', zero_division=0)


class Precision(_Loss):
    '''
    Calculate the precision score.
    Precision is also called positive predictive value.
    It is the fraction of relevant instances among the retrieved instances.
    '''
    def __init__(self, classes: int, epsilon: float = 1e-12):
        super().__init__(name="Precision", iso="prec", minimum=0.0, maximum=1.0, minimize=False)
        if not isinstance(classes, int):
            raise TypeError(f"the 'classes' specified was of wrong type {type(classes)}, expected {int}.")
        if not isinstance(epsilon, float):
            raise TypeError(f"the 'epsilon' specified was of wrong type {type(epsilon)}, expected {float}.")
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # convert from class probability to class labels
        y_pred = torch.argmax(y_pred, 1)
        return precision_score(y_true=y_true.cpu(), y_pred=y_pred.cpu(), labels=list(range(self.classes)), average='macro', zero_division=0)