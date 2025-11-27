import numpy as np
import struct
import os

SPARSITY = 0.1
SIZE = 1024
FILE_NAME = "a_matrix.bin"

FILE_DIR = "/root/workspace/samsungstf/benchmark_results"


# Generate a random matrix with target sparsity
# Save the matrix to a binary file
def gen_matrix():
    matrix = np.random.rand(SIZE, SIZE)
    matrix[matrix < SPARSITY] = 0

    with open(os.path.join(FILE_DIR, FILE_NAME), "wb") as f:
        # write the size of matrix
        f.write(struct.pack("i", SIZE))
        for i in range(SIZE):
            for j in range(SIZE):
                # float32
                f.write(struct.pack("f", matrix[i][j]))


gen_matrix()

FILE_NAME = "b_matrix.bin"
gen_matrix()
