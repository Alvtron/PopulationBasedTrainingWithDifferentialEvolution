
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

def unwrap_iterable(iterable):
    elements = list()
    unwrapped_list = iterable.values() if isinstance(iterable, dict) else iterable
    for value in unwrapped_list:
        if isinstance(value, (dict, list, tuple)):
            elements = elements + unwrap_iterable(value)
        else:
            elements.append(value)
    return elements

def merge_dict(dict1, dict2):
   ''' Merge dictionaries and keep values of common keys in list'''
   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = [value , dict1[key]]
   return dict3