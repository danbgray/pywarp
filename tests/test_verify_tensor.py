import numpy as np
import warnings
import pytest
from warp.solver.verify_tensor import verify_tensor

def test_verify_tensor_valid():
    """verify_tensor should return True for a well-formed tensor."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        input_tensor = {
            'type': "Metric",
            'tensor': np.zeros((4, 4, 2, 2, 2, 2)),
            'coords': "cartesian",
            'index': "covariant"
        }

        assert verify_tensor(input_tensor, suppress_msgs=False)
        assert verify_tensor(input_tensor, suppress_msgs=True)


def test_verify_tensor_missing_type():
    """verify_tensor should return False if the type key is missing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        input_tensor = {
            'type': "Metric",
            'tensor': np.zeros((4, 4, 2, 2, 2, 2)),
            'coords': "cartesian",
            'index': "covariant"
        }
        input_tensor.pop('type')

        assert not verify_tensor(input_tensor, suppress_msgs=False)

