import numpy as np
from typing import Dict, Any

def verify_tensor(tensor: Dict[str, Any], suppress_msgs: bool = False) -> bool:
    """Basic validation that ``tensor`` contains expected fields."""
    required_keys = {"type", "index", "tensor"}
    if not required_keys.issubset(tensor.keys()):
        return False

    if tensor["type"] not in {"metric", "energy", "stress-energy"}:
        return False

    if not isinstance(tensor["tensor"], np.ndarray):
        return False

    return True
