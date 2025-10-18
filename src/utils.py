from typing import Any


def get_first_value(d: dict, keys: list, default=None) -> Any:
    """Get the first non-None value from dict for given keys."""
    value = None
    for key in keys:
        if value := d.get(key):
            break
    return value or default
