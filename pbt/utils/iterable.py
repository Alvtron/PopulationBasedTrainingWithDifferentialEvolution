import itertools
import random
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Iterable, Iterator, Dict, Sequence, Tuple, Generator, Callable, Union, List

class Comparable(metaclass=ABCMeta):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...

T = TypeVar('T')
CT = TypeVar('CT', bound=Comparable)

def random_from_dict(values: Dict[object, CT], k: int = 1, exclude: Sequence[CT] = []) -> Tuple[CT, ...]:
    if not isinstance(exclude, Sequence):
        exclude = [exclude]
    filtered = values.values() if not exclude else [v for id, v in values.values() if v not in exclude]
    return random.sample(filtered, k) if k and k > 1 else random.choice(filtered)

def random_from_list(values: Iterable[CT], k: int = 1, exclude: Sequence[CT] = []) -> Tuple[CT, ...]:
    if not isinstance(exclude, Sequence):
        exclude = [exclude]
    filtered = values if not exclude else [v for v in values if v not in exclude]
    return random.sample(filtered, k) if k and k > 1 else random.choice(filtered)

def grid(span, n_grids):
    step = span / (n_grids + 1)
    return tuple(step * (i + 1) for i in range(n_grids))

def average(iterable: Iterable):
    total = 0
    n_values = 0
    for value in iterable:
        total += value
        n_values += 1
    return total / n_values

def split_number_evenly(number, n) -> list:
    parts = int(number / n)
    rest = number % n
    return [parts] * (n-rest) + [parts+1] * rest

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

def unwrap_iterable(iterable: Iterable[Union[Iterable, T]], exceptions: Sequence = []) -> Generator[T, None, None]:
    for value in iterable.values() if isinstance(iterable, dict) else iterable:
        if isinstance(value, Iterable) and not type(value) not in exceptions:
            yield from unwrap_iterable(value)
        else:
            yield value

def modify_iterable(iterable: Union[List[T], Dict[object,T]], expression: Callable[[T], T], condition: Callable[[T], bool] = None) -> None:
    if not isinstance(iterable, (list, dict)):
        raise TypeError("iterable must be a dict or list.")
    for key, value in iterable.items() if isinstance(iterable, dict) else enumerate(iterable):
        if isinstance(value, (list, dict)):
            modify_iterable(value, expression, condition)
        elif condition is None or condition(value):
            iterable[key] = expression(value)
        else:
            continue

def merge_dictionaries(dicts) -> defaultdict:
    ''' Merge dictionaries and keep values of common keys in list'''
    keys = set(itertools.chain(*dicts))
    result = defaultdict(list)
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

def value_by_fraction(sequence: Sequence[T], fraction: float) -> T:
    """Returns the value that reside within a specific fraction of the sequence."""
    if not sequence:
        raise ValueError("the provided sequence cannot be empty")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("the provided fraction must be a float between 0.0 and 1.0")
    index = int(fraction / (1 / len(sequence) + 1e-9))
    return sequence[index]

def singular(iterable: Iterable) -> bool:
    return len(set(iterable)) <= 1