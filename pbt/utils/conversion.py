import json

def dict_to_binary(dictionary):
    """Convert a dictionary object into bytes using encode() and json.dumps()."""
    return json.dumps(dictionary).encode('utf-8')

def binary_to_dict(binary):
    """Convert bytes into a dictionary object using decode() and json.loads(). """
    return json.loads(binary.decode('utf-8'))