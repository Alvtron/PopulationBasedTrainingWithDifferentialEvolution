from datetime import datetime
from collections import Iterable

def get_datetime_string():
    date_and_time = datetime.now()
    return date_and_time.strftime('%Y-%m-%d %H:%M:%S')

def unwrap_iterable(iterable):
    elements = []
    list = iterable.values() if isinstance(iterable, dict) else iterable
    for value in list:
        if isinstance(value, Iterable):
            elements = elements + unwrap_iterable(value)
        else:
            elements.append(value)
    return elements
            