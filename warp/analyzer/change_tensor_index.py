import numpy as np
from typing import Dict, Any, Union

def change_tensor_index(input_tensor: Dict[str, Any], index: str, metric_tensor: Union[Dict[str, Any], None] = None) -> Dict[str, Any]:
    """
    Changes a tensor's index.

    Args:
        input_tensor (dict): Tensor struct to change the index of.
        index (str): Index to change the input_tensor to, such as 'covariant', 'contravariant', 'mixedupdown', 'mixeddownup'.
        metric_tensor (dict, optional): Metric struct. Required if input_tensor is not of type 'metric'.

    Returns:
        dict: Tensor struct in the provided index.

    Raises:
        ValueError: If metric_tensor is needed but not provided or if an invalid transformation is selected.
    """

    if metric_tensor is None:
        if input_tensor['type'].lower() != 'metric':
            raise ValueError("metricTensor is needed as third input when changing index of non-metric tensors.")
    else:
        if metric_tensor['index'].lower() in ['mixedupdown', 'mixeddownup']:
            raise ValueError("Metric tensor cannot be used in mixed index.")

    if index.lower() not in ['covariant', 'contravariant', 'mixedupdown', 'mixeddownup']:
        raise ValueError('Transformation selected is not allowed, use either: "covariant", "contravariant", "mixedupdown", "mixeddownup"')

    output_tensor = input_tensor.copy()
    if input_tensor['type'].lower() == 'metric':
        if (input_tensor['index'].lower() == 'covariant' and index.lower() == 'contravariant') or (input_tensor['index'].lower() == 'contravariant' and index.lower() == 'covariant'):
            output_tensor['tensor'] = np.linalg.inv(input_tensor['tensor'])
        elif input_tensor['index'].lower() in ['mixedupdown', 'mixeddownup']:
            raise ValueError("Input tensor is a Metric tensor of mixed index.")
        elif index.lower() in ['mixedupdown', 'mixeddownup']:
            raise ValueError("Cannot convert a metric tensor to mixed index.")
    else:
        if input_tensor['index'].lower() == 'covariant' and index.lower() == 'contravariant':
            if metric_tensor['index'].lower() == 'covariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'contravariant'
            output_tensor['tensor'] = flip_index(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'contravariant' and index.lower() == 'covariant':
            if metric_tensor['index'].lower() == 'contravariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'covariant'
            output_tensor['tensor'] = flip_index(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'contravariant' and index.lower() == 'mixedupdown':
            if metric_tensor['index'].lower() == 'contravariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'covariant'
            output_tensor['tensor'] = mix_index2(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'contravariant' and index.lower() == 'mixeddownup':
            if metric_tensor['index'].lower() == 'contravariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'covariant'
            output_tensor['tensor'] = mix_index1(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'covariant' and index.lower() == 'mixedupdown':
            if metric_tensor['index'].lower() == 'covariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'contravariant'
            output_tensor['tensor'] = mix_index1(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'covariant' and index.lower() == 'mixeddownup':
            if metric_tensor['index'].lower() == 'covariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'contravariant'
            output_tensor['tensor'] = mix_index2(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'mixedupdown' and index.lower() == 'contravariant':
            if metric_tensor['index'].lower() == 'covariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'contravariant'
            output_tensor['tensor'] = mix_index2(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'mixedupdown' and index.lower() == 'covariant':
            if metric_tensor['index'].lower() == 'contravariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'covariant'
            output_tensor['tensor'] = mix_index1(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'mixeddownup' and index.lower() == 'covariant':
            if metric_tensor['index'].lower() == 'contravariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'covariant'
            output_tensor['tensor'] = mix_index2(input_tensor, metric_tensor)
        elif input_tensor['index'].lower() == 'mixeddownup' and index.lower() == 'contravariant':
            if metric_tensor['index'].lower() == 'covariant':
                metric_tensor['tensor'] = np.linalg.inv(metric_tensor['tensor'])
                metric_tensor['index'] = 'contravariant'
            output_tensor['tensor'] = mix_index1(input_tensor, metric_tensor)

    output_tensor['index'] = index
    return output_tensor

def flip_index(input_tensor: Dict[str, Any], metric_tensor: Dict[str, Any]) -> np.ndarray:
    """
    Flips the index of a tensor using the metric tensor.

    Args:
        input_tensor (dict): Tensor struct.
        metric_tensor (dict): Metric tensor struct.

    Returns:
        np.ndarray: Tensor with flipped index.
    """
    temp_output_tensor = np.zeros_like(input_tensor['tensor'])
    for i in range(4):
        for j in range(4):
            for a in range(4):
                for b in range(4):
                    temp_output_tensor[i, j] += (input_tensor['tensor'][a, b] *
                                                 metric_tensor['tensor'][a, i] *
                                                 metric_tensor['tensor'][b, j])
    return temp_output_tensor

def mix_index1(input_tensor: Dict[str, Any], metric_tensor: Dict[str, Any]) -> np.ndarray:
    """
    Mixes the index of a tensor using the metric tensor (method 1).

    Args:
        input_tensor (dict): Tensor struct.
        metric_tensor (dict): Metric tensor struct.

    Returns:
        np.ndarray: Tensor with mixed index.
    """
    temp_output_tensor = np.zeros_like(input_tensor['tensor'])
    for i in range(4):
        for j in range(4):
            for a in range(4):
                temp_output_tensor[i, j] += input_tensor['tensor'][a, j] * metric_tensor['tensor'][a, i]
    return temp_output_tensor

def mix_index2(input_tensor: Dict[str, Any], metric_tensor: Dict[str, Any]) -> np.ndarray:
    """
    Mixes the index of a tensor using the metric tensor (method 2).

    Args:
        input_tensor (dict): Tensor struct.
        metric_tensor (dict): Metric tensor struct.

    Returns:
        np.ndarray: Tensor with mixed index.
    """
    temp_output_tensor = np.zeros_like(input_tensor['tensor'])
    for i in range(4):
        for j in range(4):
            for a in range(4):
                temp_output_tensor[i, j] += input_tensor['tensor'][i, a] * metric_tensor['tensor'][a, j]
    return temp_output_tensor

def c4_inv(tensor: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a 4x4 tensor.

    Args:
        tensor (np.ndarray): 4x4 tensor.

    Returns:
        np.ndarray: Inverse of the tensor.
    """
    return np.linalg.inv(tensor)
