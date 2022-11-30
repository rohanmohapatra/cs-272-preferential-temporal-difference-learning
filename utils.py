from typing import Any


def argmax_action(d: dict[Any, float]) -> Any:
    """return a key of the maximum value in a given dictionary 

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """
    return max(d, key=d.get)
