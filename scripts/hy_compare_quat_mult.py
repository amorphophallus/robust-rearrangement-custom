import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from furniture_bench.controllers.control_utils import quat_multiply
from src.common.geometry import np_apply_quat

def compare_quat_functions():
    # Generate random quaternions for testing
    np.random.seed(42)
    q1 = np.random.rand(4) - 0.5  # Random quaternion
    q2 = np.random.rand(4) - 0.5  # Random quaternion

    # Normalize the quaternions
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    # Compute output using np_apply_quat
    np_output = np_apply_quat(q1, q2)

    # Compute output using quat_multiply
    q1_torch = torch.tensor(q1, dtype=torch.float32)
    q2_torch = torch.tensor(q2, dtype=torch.float32)
    torch_output = quat_multiply(q1_torch, q2_torch).numpy()

    # Compare the outputs
    print("Input Quaternion 1 (q1):", q1)
    print("Input Quaternion 2 (q2):", q2)
    print("Output from np_apply_quat:", np_output)
    print("Output from quat_multiply:", torch_output)

    # Check if the outputs are close
    if np.allclose(np_output, torch_output, atol=1e-6):
        print("The outputs are the same!")
    else:
        print("The outputs are different!")

if __name__ == "__main__":
    compare_quat_functions()