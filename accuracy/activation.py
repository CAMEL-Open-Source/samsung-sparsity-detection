import torch

# Define matrix dimensions
A, B = 10000, 10000

# Ensure we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create random tensor on GPU
x = torch.randn(A, B, device=device)

# Define activation functions
relu = torch.nn.ReLU()
gelu = torch.nn.GELU()


# Function to measure time using CUDA events
def measure_time(activation_func, x):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record start
    start_event.record()

    # Run the activation
    for _ in range(10):
        _ = activation_func(x)

    # Record end
    end_event.record()

    # Wait for everything to finish
    torch.cuda.synchronize()

    # Return elapsed time in milliseconds
    return start_event.elapsed_time(end_event)


# Warm-up GPU (optional but recommended for more stable timing)
for _ in range(10):
    _ = relu(x)
    _ = gelu(x)

# Measure execution time for ReLU
time_relu = measure_time(relu, x)
print(f"Time for ReLU: {time_relu:.3f} ms")

# Measure execution time for GeLU
time_gelu = measure_time(gelu, x)
print(f"Time for GeLU: {time_gelu:.3f} ms")
