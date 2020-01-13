import torch
import torch.nn.functional as F
import collections

class _Loss(torch.nn.Module):
    '''
    Base class for torch loss functions.
    Requires specification of in which direction the score is considered better.
    '''
    def __init__(self, name, iso, min, max, minimize):
        super().__init__()
        self.name = name
        self.iso = iso
        self.min = min
        self.max = max
        self.minimize = minimize

    def forward(self, y_pred, y_true):
        raise NotImplementedError()

class Accuracy(_Loss):
    '''
    Calculate the accuracy.
    '''
    def __init__(self, in_decimal=False):
        super().__init__(
            "Accuracy", "acc",
            min=0.0, max=1.0 if in_decimal else 100.0, minimize=False)
        self.in_decimal = in_decimal
    
    def forward(self, y_pred, y_true):
        n_samples = torch.numel(y_true)
        predicted = torch.argmax(y_pred, dim=1)
        n_correct = predicted.eq(y_true).sum().float()
        accuracy = n_correct / float(n_samples)
        return accuracy if self.in_decimal else accuracy * 100.0

class MAE(_Loss):
    '''
    Calculate the mean absolute error (L1 norm).
    '''
    def __init__(self, reduction='mean'):
        super().__init__(
            "Mean Absolute Error", "l1",
            min=0, max=None, minimize=True)
        self.function = torch.nn.L1Loss(
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)

class MSE(_Loss):
    '''
    Calculate the mean squared error (squared L2 norm).
    '''
    def __init__(self, reduction='mean'):
        super().__init__(
            "Mean Squared Error", "l2",
            min=0, max=None, minimize=True)
        self.function = torch.nn.MSELoss(
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)

class CategoricalCrossEntropy(_Loss):
    '''
    Calculate the cross-entropy loss.
    '''
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(
            "Categorical Cross Entropy", "cce",
            min=0, max=None, minimize=True)
        self.function = torch.nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)

class BinaryCrossEntropy(_Loss):
    '''
    Calculate the binary cross-entropy loss.
    '''
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(
            "Categorical Cross Entropy", "cce",
            min=0, max=None, minimize=True)
        self.function = torch.nn.BCELoss(
            weight=weight,
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)

class NLL(_Loss):
    '''
    Calculate the negative log likelihood loss.
    '''
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(
            "Negative Log Likelihood", "nll",
            min=0, max=None, minimize=True)
        self.function = torch.nn.NLLLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)    

class F1(_Loss):
    '''
    Calculate the F1 score.
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__(
            "F1 Score", "f1",
            min=0, max=1.0, minimize=False)
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2
        if y_pred.ndim == 2:
            y_pred = torch.argmax(y_pred, dim=1)
        tp = (y_true * y_pred).sum().to(torch.float32)
        #tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        return f1
    