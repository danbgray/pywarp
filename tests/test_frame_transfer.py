import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warp.analyzer.do_frame_transfer import do_frame_transfer


def test_do_frame_transfer_basic():
    metric = {
        'type': 'metric',
        'index': 'covariant',
        'tensor': np.eye(4).reshape(4,4,1,1),
        'coords': 'cartesian'
    }
    energy = {
        'type': 'energy',
        'index': 'contravariant',
        'tensor': np.zeros((4,4,1,1))
    }
    transformed = do_frame_transfer(metric, energy, 'Eulerian', 0)
    assert transformed['frame'] == 'Eulerian'
    assert transformed['index'] == 'contravariant'
