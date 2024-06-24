import numpy as np
import warnings
from warp.solver.verify_tensor import verify_tensor

def test_verify_tensor():
    # Suppress specific warnings during tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        # Example tensor with correct properties
        input_tensor = {
            'type': "Metric",
            'tensor': np.zeros((4, 4, 2, 2, 2)),
            'coords': "cartesian",
            'index': "covariant"
        }

        # Test with suppress_msgs = False
        print("Test 1: Expected output - All verifications pass")
        verified = verify_tensor(input_tensor, suppress_msgs=False)
        print("Verified:", verified)

        # Test with suppress_msgs = True
        print("\nTest 2: Expected output - No messages")
        verified = verify_tensor(input_tensor, suppress_msgs=True)
        print("Verified:", verified)

        # Test with missing type
        print("\nTest 3: Expected output - Warning for missing type")
        input_tensor.pop('type')
        verified = verify_tensor(input_tensor, suppress_msgs=False)
        print("Verified:", verified)

if __name__ == "__main__":
    test_verify_tensor()
