import numpy as np
import warnings

def verify_tensor(input_tensor, suppress_msgs=False):
    """
    Verifies the metric tensor and stress energy tensor properties.

    Args:
    - input_tensor (dict): Tensor to verify.
    - suppress_msgs (bool, optional): Flag to suppress messages. Defaults to False.

    Returns:
    - verified (bool): Whether the tensor is verified or not.
    """
    def disp_message(msg, sM):
        if not sM:
            print(msg)

    verified = True

    # Check if type field exists
    if 'type' in input_tensor:
        tensor_type = input_tensor['type'].lower()
        if tensor_type == "metric":
            disp_message("type: Metric", suppress_msgs)
        elif tensor_type == "stress-energy":
            disp_message("type: Stress-Energy", suppress_msgs)
        else:
            warnings.warn("Unknown type")
            verified = False
    else:
        warnings.warn('Tensor type does not exist. Must be either "Metric" or "Stress-Energy"')
        verified = False

    # Check other properties
    # Tensor
    if 'tensor' in input_tensor:
        tensor = input_tensor['tensor']
        if isinstance(tensor, np.ndarray) and tensor.shape == (4, 4, *tensor.shape[2:]) and tensor.ndim == 4:
            disp_message("tensor: Verified", suppress_msgs)
        else:
            warnings.warn("Tensor is not formatted correctly. Tensor must be a 4x4 array of 4D values.")
            verified = False
    else:
        warnings.warn("tensor: Empty")
        verified = False

    # Coords
    if 'coords' in input_tensor:
        coords = input_tensor['coords'].lower()
        if coords == "cartesian":
            disp_message("coords: " + coords, suppress_msgs)
        else:
            warnings.warn("Non-cartesian coordinates are not supported at this time. Set .coords to 'cartesian'.")
            verified = False
    else:
        warnings.warn("coords: Empty")
        verified = False

    # Index
    if 'index' in input_tensor:
        index = input_tensor['index'].lower()
        if index in ["contravariant", "covariant", "mixedupdown", "mixeddownup"]:
            disp_message("index: " + index, suppress_msgs)
        else:
            warnings.warn("Unknown index")
            verified = False
    else:
        warnings.warn("index: Empty")
        verified = False

    return verified
