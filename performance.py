import torch

def accuracy(output, y):
    n_samples = torch.numel(y) # number of samples
    predicted = torch.argmax(output, dim=1) # get predictions
    n_correct = predicted.eq(y).sum().float() # how many are correct
    return (n_correct / float(n_samples)) * 100.