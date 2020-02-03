import torch

class _Loss(torch.nn.Module):
    '''
    Base class for torch loss functions, with extra information.
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
        super().__init__("Accuracy", "acc", min=0.0, max=1.0 if in_decimal else 100.0, minimize=False)
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
        super().__init__("Mean Absolute Error", "l1", min=0, max=None, minimize=True)
        self.function = torch.nn.L1Loss(
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)

class MSE(_Loss):
    '''
    Calculate the mean squared error (squared L2 norm).
    '''
    def __init__(self, reduction='mean'):
        super().__init__("Mean Squared Error", "l2", min=0, max=None, minimize=True)
        self.function = torch.nn.MSELoss(
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)

class CategoricalCrossEntropy(_Loss):
    '''
    Calculate the cross-entropy loss.
    '''
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__("Categorical Cross Entropy", "cce", min=0, max=None, minimize=True)
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
        super().__init__("Categorical Cross Entropy", "cce", min=0, max=None, minimize=True)
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
        super().__init__("Negative Log Likelihood", "nll", min=0, max=None, minimize=True)
        self.function = torch.nn.NLLLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction)
        
    def forward(self, y_pred, y_true):
        return self.function(y_pred, y_true)    

def confusion_matrix(y_pred, y_true, n_classes):
    conf_matrix = torch.zeros(n_classes, n_classes)
    y_pred = torch.argmax(y_pred, 1)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    return conf_matrix

class F1(_Loss):
    '''
    Calculate the F1 score.
    F1 score is also known as balanced F-score or F-measure.
    It is the weighted average of the precision and sensitivity (recall).
    '''
    def __init__(self, classes, epsilon=1e-7):
        super().__init__("F1 Score", "f1", min=0, max=1.0, minimize=False)
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        conf_matrix = confusion_matrix(y_pred, y_true, self.classes)
        TP = conf_matrix.diag()
        sensitivity = 0.0
        precision = 0.0
        for label in range(self.classes):
            idx = torch.ones(self.classes).long()
            idx[label] = 0
            FP = conf_matrix[label, idx].sum()
            FN = conf_matrix[idx, label].sum()
            sensitivity += (TP[label] / (TP[label] + FN + self.epsilon)) / self.classes
            precision += (TP[label] / (TP[label] + FP + self.epsilon)) / self.classes
        f1_score = 2.0 * ((precision * sensitivity) / (precision + sensitivity + self.epsilon))
        return f1_score

class Sensitivity(_Loss):
    '''
    Calculate the sensitivity score.
    Sensitivity is also called the true positive rate, the recall, or probability of detection.
    It is the fraction of the total amount of relevant instances that were actually retrieved.
    '''
    def __init__(self, classes, epsilon=1e-7):
        super().__init__("Sensitivity", "sensitivity", min=0, max=1.0, minimize=False)
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        conf_matrix = confusion_matrix(y_pred, y_true, self.classes)
        TP = conf_matrix.diag()
        sensitivity = 0.0
        for label in range(self.classes):
            idx = torch.ones(self.classes).byte()
            idx[label] = 0
            FN = conf_matrix[idx, label].sum()
            sensitivity += (TP[label] / (TP[label] + FN + self.epsilon)) / self.classes
        return sensitivity

class Specificity(_Loss):
    '''
    Calculate the specificity score.
    Specificity is also called the true negative rate.
    It measures the proportion of actual negatives that are correctly identified.
    '''
    def __init__(self, classes, epsilon=1e-7):
        super().__init__("Specificity", "specificity", min=0, max=1.0, minimize=False)
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        conf_matrix = confusion_matrix(y_pred, y_true, self.classes)
        specificity = 0.0
        for label in range(self.classes):
            idx = torch.ones(self.classes).byte()
            idx[label] = 0
            TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
            FP = conf_matrix[label, idx].sum()
            specificity += (TN / (TN + FP + self.epsilon)) / self.classes
        return specificity

class Precision(_Loss):
    '''
    Calculate the precision score.
    Precision is also called positive predictive value.
    It is the fraction of relevant instances among the retrieved instances.
    '''
    def __init__(self, classes, epsilon=1e-7):
        super().__init__("Precision", "precision", min=0, max=1.0, minimize=False)
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        conf_matrix = confusion_matrix(y_pred, y_true, self.classes)
        TP = conf_matrix.diag()
        precision = 0.0
        for label in range(self.classes):
            idx = torch.ones(self.classes).byte()
            idx[label] = 0
            FP = conf_matrix[label, idx].sum()
            precision += (TP[label] / (TP[label] + FP + self.epsilon)) / self.classes
        return precision