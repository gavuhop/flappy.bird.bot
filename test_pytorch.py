import torch
import numpy as np


def test_pytorch():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    # Create a simple tensor
    x = torch.randn(5, 3)
    print("\nRandom tensor:")
    print(x)

    # Perform a simple operation
    y = x + 1
    print("\nTensor after adding 1:")
    print(y)

    # Convert between numpy and torch
    np_array = np.array([[1, 2, 3], [4, 5, 6]])
    torch_tensor = torch.from_numpy(np_array)
    print("\nNumpy array converted to torch tensor:")
    print(torch_tensor)

    # Test neural network
    model = torch.nn.Linear(3, 2)
    input_tensor = torch.randn(2, 3)
    output = model(input_tensor)
    print("\nNeural network output:")
    print(output)

    print("\nPyTorch test completed successfully!")


if __name__ == "__main__":
    test_pytorch()
