import math

def translate(value, left_min, left_max, right_min, right_max):
    # Calculate the span of each range
    left_span = left_max - left_min
    right_span = right_max - right_min
    # normalize the value from the left range into a float between 0 and 1
    value_normalized = float(value - left_min) / float(left_span)
    # Convert the normalize value range into a value in the right range.
    return right_min + (value_normalized * right_span)

def clip(value, min_value, max_value):
    if value <= min_value:
        return min_value
    elif value >= max_value:
        return max_value
    else:
        return value

def reflect(value, min_value, max_value):
    span = max_value - min_value
    if value < min_value:
        length = min_value - value
        times = math.floor(length / span)
        delta = length - span * times
        return min_value + delta if times % 2 == 0 else max_value - delta
    elif value > max_value:
        length = value - max_value
        times = math.floor(length / span)
        delta = length - span * times
        return max_value - delta if times % 2 == 0 else min_value + delta
    else:
        return value

def reflect_recursive(value, min_value, max_value):
    if value < min_value:
        return reflect_recursive(min_value + (min_value - value), min_value, max_value)
    elif value > max_value:
        return reflect_recursive(max_value - (value - max_value), min_value, max_value)
    else:
        return value