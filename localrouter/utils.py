import json
import os


def load_json(path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary."""
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        data = json.load(f)
    return data


def dict_recursive(
    condition=lambda x: not isinstance(x, dict) and not isinstance(x, list)
):
    """
    Creates a decorotor that applies a function recursively to all elements of a dict or list.
    Recognizes leaf nodes by the condition function.
    """

    def decorator(f):
        def wrapped(data, *args, **kwargs):
            if condition(data):
                return f(data, *args, **kwargs)
            if isinstance(data, dict):
                return {k: wrapped(v, *args, **kwargs) for k, v in data.items()}
            if isinstance(data, list):
                return [wrapped(v, *args, **kwargs) for v in data]
            return data

        return wrapped

    return decorator
