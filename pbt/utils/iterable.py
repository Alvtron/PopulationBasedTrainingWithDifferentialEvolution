import itertools
import random
import collections
from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Iterable, Dict, Sequence, Tuple

class Comparable(metaclass=ABCMeta):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...

CT = TypeVar('CT', bound=Comparable)

def random_from_dict(members : Dict[object, CT], k : int = 1, exclude : Sequence[CT] = None) -> Tuple[CT, ...]:
    if not isinstance(exclude, collections.Sequence):
        exclude = [exclude]
    filtered = members.values() if not exclude else [m for id, m in members.values() if m not in exclude]
    return random.sample(filtered, k) if k and k > 1 else random.choice(filtered)

def random_from_list(members : Iterable[CT], k : int = 1, exclude : Sequence[CT] = None) -> Tuple[CT, ...]:
    if not isinstance(exclude, collections.Sequence):
        exclude = [exclude]
    filtered = members if not exclude else [m for m in members if m not in exclude]
    return random.sample(filtered, k) if k and k > 1 else random.choice(filtered)

def grid(span, n_grids):
    step = span / (n_grids + 1)
    return tuple(step * (i + 1) for i in range(n_grids))

def average(iterable : Iterable):
    sum = 0
    n_values = 0
    for value in iterable:
        sum += value
        n_values += 1
    return sum / n_values

def split_number(x, n):
    """Split the number into N parts such that difference between the smallest and the largest part is minimum."""
    # If we cannot split the number into exactly 'N' parts 
    if(x < n):
        raise ValueError
    # If x % n == 0 then the minimum difference is 0 and all numbers are x / n 
    elif (x % n == 0):
        for i in range(n):
            yield x / n 
    else:
        # upto n-(x % n) the values will be x / n  
        # after that the values will be x / n + 1 
        zp = n - (x % n)
        pp = x / n
        for i in range(n):
            if(i >= zp):
                yield pp + 1
            else:
                yield pp

def flatten_dict(dictionary, exclude = [], delimiter ='_'):
    flat_dict = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict) and key not in exclude:
            flatten_value_dict = flatten_dict(value, exclude, delimiter)
            for k, v in flatten_value_dict.items():
                flat_dict[f"{key}{delimiter}{k}"] = v
        else:
            flat_dict[key] = value
    return flat_dict

def unwrap_iterable(iterable) -> list:
    elements = list()
    unwrapped_list = iterable.values() if isinstance(iterable, dict) else iterable
    for value in unwrapped_list:
        if isinstance(value, (dict, list, tuple)):
            elements = elements + unwrap_iterable(value)
        else:
            elements.append(value)
    return elements

def merge_dictionaries(dicts) -> collections.defaultdict:
    ''' Merge dictionaries and keep values of common keys in list'''
    keys = set(itertools.chain(*dicts))
    result = collections.defaultdict(list)
    for d in dicts:
        for key in keys:
            if key in d:
                result[key].append(d[key])
    return result

def chunks(sequence, n):
    """Return a generator that yields successive n-sized chunks from a sequence."""
    for i in range(0, len(sequence), n):
        yield sequence[i:i + n]

def insert_sequence(index, seq1, seq2):
    """Inserts the second sequence on the index in the first sequence."""
    return seq1[:index] + seq2 + seq1[index:]