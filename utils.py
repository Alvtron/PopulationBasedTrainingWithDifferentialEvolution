import math
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

def translate(value, left_min, left_max, right_min, right_max):
    # Calculate the span of each range
    left_span = left_max - left_min
    right_span = right_max - right_min
    # normalize the value from the left range into a float between 0 and 1
    value_normalized = float(value - left_min) / float(left_span)
    # Convert the normalize value range into a value in the right range.
    return right_min + (value_normalized * right_span)