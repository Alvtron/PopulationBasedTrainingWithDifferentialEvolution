import torch
import torch.nn.functional as F

class _Loss(torch.nn.Module):
    '''
    Base class for torch loss functions.
    Requires specification of in which direction the score is considered better.
    '''
    def __init__(self, minimize):
        super().__init__()
        self.minimize = minimize

    def forward(self, y_pred, y_true):
        raise NotImplementedError()

class Accuracy(_Loss):
    '''
    Calculate the accuracy.
    '''
    def __init__(self, in_decimal=False):
        super().__init__(False)
        self.in_decimal = in_decimal
    
    def forward(self, y_pred, y_true):
            n_samples = torch.numel(y_true)
            predicted = torch.argmax(y_pred, dim=1)
            n_correct = predicted.eq(y_true).sum().float()
            accuracy = n_correct / float(n_samples)
            if not self.in_decimal:
                accuracy *= 100.
            return accuracy 

class MAE(_Loss):
    '''
    Calculate the mean absolute error (L1 norm).
    '''
    def __init__(self, reduction='mean'):
        super().__init__(True)
        self.function = torch.nn.L1Loss(
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)

class MSE(_Loss):
    '''
    Calculate the mean squared error (squared L2 norm).
    '''
    def __init__(self, reduction='mean'):
        super().__init__(True)
        self.function = torch.nn.MSELoss(
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)

class CategoricalCrossEntropy(_Loss):
    '''
    Calculate the cross-entropy loss.
    '''
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(True)
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
        super().__init__(True)
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
        super().__init__(True)
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
        super().__init__(True)
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).float()
        y_pred = F.softmax(y_pred, dim=1)
        # calc true positives, false positives and false negatives
        tp = (y_true * y_pred).sum(dim=0).float()
        fp = ((1 - y_true) * y_pred).sum(dim=0).float()
        fn = (y_true * (1 - y_pred)).sum(dim=0).float()
        # calc precision and recall
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        # calc f1
        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()
    