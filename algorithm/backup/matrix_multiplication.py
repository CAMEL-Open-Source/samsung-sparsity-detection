import numpy as np
import struct
import os
import torch
import time
import argparse
import json


class GNNConfig:
    def __init__(self):
        # Constants
        self.SPARSITY = 0.95
        self.SIZE = 4096
        self.N_TESTS = 10
        self.DEVICE = "cuda:0"
        self.OUTPUT_FILE_PATH = "config.json"

        self.format_conversion_time = []
        self.matrix_multiplication_time = []

    def overide_config_from_json(self, json_config):
        for key, value in json_config.items():
            # check key is capital
            if key.isupper():
                setattr(self, key, value)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="GNN Configuration")
        for key, value in self.__dict__.items():
            if key.isupper():
                if isinstance(value, bool):
                    parser.add_argument(f"--{key}", type=str, help=key)
                else:
                    parser.add_argument(f"--{key}", type=type(value), help=key)

        args = parser.parse_args()
        for key, value in vars(args).items():
            if value is not None:
                if value == "True":
                    value = True
                elif value == "False":
                    value = False
                setattr(self, key, value)

    def post_process(self):
        config.format_conversion_time = np.average(config.format_conversion_time)
        config.matrix_multiplication_time = np.average(
            config.matrix_multiplication_time
        )

    def save_config(self):
        self.post_process()
        with open(self.OUTPUT_FILE_PATH, "w") as file:
            json.dump(self.__dict__, file, indent=4)


config = GNNConfig()
config.parse_args()

matrix_a = np.random.rand(config.SIZE, config.SIZE)
matrix_b = np.random.rand(config.SIZE, config.SIZE)

matrix_a[matrix_a < config.SPARSITY] = 0
matrix_b[matrix_b < config.SPARSITY] = 0

matrix_a = torch.tensor(matrix_a).to(config.DEVICE)
matrix_b = torch.tensor(matrix_b).to(config.DEVICE)

for _ in range(config.N_TESTS + 5):
    start = time.time()
    # convert to sparse matrix
    matrix_a_sparse = matrix_a.to_sparse()
    matrix_b_sparse = matrix_b.to_sparse()
    config.format_conversion_time.append(time.time() - start)

    start = time.time()
    # matrix multiplication
    # torch.sparse.mm(matrix_a_sparse, matrix_b_sparse)
    torch.mm(matrix_a, matrix_b)
    config.matrix_multiplication_time.append(time.time() - start)

config.save_config()
