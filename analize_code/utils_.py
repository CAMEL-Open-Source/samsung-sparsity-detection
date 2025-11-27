import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys
import json
import random
import subprocess
import struct
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np


class MatMulConfig:
    def __init__(self):
        # Constants
        self.DETECT_PERIOD = 10
        self.DETECT_METHOD = "accurate"  # accurate or sparsity_based or weight_based

        # cpu or cuda:0
        self.DEVICE = "cuda:0"
        self.OUTPUT_FILE_PATH = "config.json"
        self.TEMP_PATH = "/root/workspace/samsungstf/benchmark_results"
        self.DEBUG_PRINT = True

        self.A_HEIGHT = 1024
        self.A_WIDTH = 1024
        self.B_HEIGHT = self.A_WIDTH
        self.B_WIDTH = 1024

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
        pass

    def save_config(self):
        self.post_process()
        with open(self.OUTPUT_FILE_PATH, "w") as file:
            json.dump(self.__dict__, file, indent=4)


models = None


def predict_time(algorithm, size, sparsityA, sparsityB):
    global models
    if models is None:
        data = []
        with open(
            f"/root/workspace/samsungstf/results/{size}_matmul_result.txt", "r"
        ) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                tokens = line.split()
                if len(tokens) != 4:
                    continue
                algorithm, sparsityA, sparsityB, time = tokens
                data.append(
                    {
                        "algorithm": algorithm,
                        "sparsityA": float(sparsityA),
                        "sparsityB": float(sparsityB),
                        "time": float(time),
                    }
                )

        df = pd.DataFrame(data)

        models = {}
        for algorithm in df["algorithm"].unique():
            df_alg = df[df["algorithm"] == algorithm]
            X = df_alg[["sparsityA", "sparsityB"]]
            y = df_alg["time"]

            model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            model.fit(X, y)

            models[algorithm] = model

    if algorithm.startswith("reverse_"):
        algorithm = algorithm[8:]
        temp = sparsityA
        sparsityA = sparsityB
        sparsityB = temp
    if algorithm not in models:
        print(f"No model available for algorithm {algorithm}")
        return None
    model = models[algorithm]
    X_new = pd.DataFrame({"sparsityA": [sparsityA], "sparsityB": [sparsityB]})
    time_pred = model.predict(X_new)
    return time_pred[0]


def make_random_sparse_matrix(height, width, sparsity):
    matrix = np.random.rand(height, width)
    matrix[matrix < sparsity] = 0
    return matrix


def get_best_algorithm(a_matrix, b_matrix, sparsity_info, config: MatMulConfig):
    alg_list = "GEMM SpMMRowR SpMMRowC SpMMElemR SpMMElemC SpGEMMHashR SpGEMMHashC SpGEMMSortR SpGEMMSortC".split()
    if True:
        alg_list_ = []
        for alg in alg_list:
            alg_list_.append("reverse_" + alg)
        alg_list += alg_list_
    if config.DETECT_METHOD == "accurate":
        time_map = {}
        for alg in alg_list:
            time_map[alg] = matrix_multiplication(a_matrix, b_matrix, alg, config)[1]
        if config.DEBUG_PRINT:
            for alg, time in time_map.items():
                print(f"### {alg}: {time:.3f} ms")
        min_time = min(time_map.values())
        return min(time_map, key=time_map.get), min_time * len(alg_list)
    elif config.DETECT_METHOD == "sparsity_based":
        if models == None:
            predict_time(alg, config.A_HEIGHT, 0.1, 0.1)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        a_matrix_torch = torch.tensor(a_matrix).to(config.DEVICE)
        b_matrix_torch = torch.tensor(b_matrix).to(config.DEVICE)
        start_time.record()
        a_matrix_csr = a_matrix_torch.to_sparse_csr()
        sparsity_a = 1 - a_matrix_csr._nnz() / a_matrix_csr.numel()
        b_matrix_csr = b_matrix_torch.to_sparse_csr()
        sparsity_b = 1 - b_matrix_csr._nnz() / b_matrix_csr.numel()
        end_time.record()
        torch.cuda.synchronize()
        detection_time = start_time.elapsed_time(end_time)
        time_map = {}
        for alg in alg_list:
            time_map[alg] = predict_time(alg, config.A_HEIGHT, sparsity_a, sparsity_b)
        if config.DEBUG_PRINT:
            for alg, time in time_map.items():
                print(f"### {alg}: {time:.3f} ms")
        min_time = min(time_map.values())
        return min(time_map, key=time_map.get), detection_time
    elif config.DETECT_METHOD == "sparsity_avg_based":
        time_map = {}
        for alg in alg_list:
            time_map[alg] = predict_time(
                alg,
                config.A_HEIGHT,
                sparsity_info["input_sparsity_avg"],
                sparsity_info["weight_sparsity_avg"],
            )
        return min(time_map, key=time_map.get), 0.001
    elif config.DETECT_METHOD == "ml_based":
        # A Pattern-Based SpGEMM Library for Multi-Core and Many-Core Architectures
        # about 37% overhead compared to exact method, 92% accuracy
        time_map = {}
        for alg in alg_list:
            time_map[alg] = matrix_multiplication(a_matrix, b_matrix, alg, config)[1]
        for alg in alg_list:
            # make random 0~1
            random_value = np.random.rand()
            if random_value < 0.5:
                time_map[alg] = time_map[alg] * 0.92
            else:
                time_map[alg] = time_map[alg] * 1.08
        if config.DEBUG_PRINT:
            for alg, time in time_map.items():
                print(f"### {alg}: {time:.3f} ms")
        min_time = min(time_map.values())
        return min(time_map, key=time_map.get), min_time * 0.37
    elif config.DETECT_METHOD == "random":
        return random.choice(alg_list), 0
    elif config.DETECT_METHOD == "fixed":
        return "GEMM", 0


def gen_matrix(matrix, file_dir, file_name):
    height, width = matrix.shape
    with open(os.path.join(file_dir, file_name), "wb") as f:
        # write the size of matrix
        f.write(struct.pack("i", height))
        for i in range(height):
            for j in range(width):
                # float32
                f.write(struct.pack("f", matrix[i][j]))


def get_format_conversion_time(config: MatMulConfig, a_matrix):
    a_matrix_torch = torch.tensor(a_matrix).to(config.DEVICE)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    a_matrix_csr = a_matrix_torch.to_sparse_csr()
    end_time.record()
    torch.cuda.synchronize()
    return start_time.elapsed_time(end_time)


transpose_time = None


def get_transpose_time(config: MatMulConfig):
    global transpose_time
    if transpose_time == None:
        a_matrix = torch.randn(config.A_HEIGHT, config.B_WIDTH).to(config.DEVICE)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        a_matrix = a_matrix.t().contiguous()
        end_time.record()
        torch.cuda.synchronize()
        transpose_time = start_time.elapsed_time(end_time)
    return transpose_time


def matrix_multiplication(
    a_matrix: float,
    b_matrix: float,
    algorithm: str,
    config: MatMulConfig,
    sparsity_info=None,
    use_prediction=False,
):
    # GEMM SpMMRowR SpMMRowC SpMMElemR SpMMElemC SpGEMMHashR SpGEMMHashC SpGEMMSortR SpGEMMSortC
    # assume input matrices are dense

    if use_prediction:
        time_pred = predict_time(
            algorithm,
            config.A_HEIGHT,
            sparsity_info["input_sparsity"],
            sparsity_info["weight_sparsity"],
        )
        if time_pred is not None:
            return [0, time_pred, 0, time_pred]

    dict_name = {
        "GEMM": "gemm_tile",
        "GEMMTile": "gemm_tile",
        "GEMMNtile": "gemm_ntile",
        "SpMMRowR": "spmm_csr_row",
        "SpMMRowC": "spmm_csr_col",
        "SpMMElemR": "spmm_csr_elem_row",
        "SpMMElemC": "spmm_csr_elem_col",
        "SpMMCusp": "spmm_cusparse",
        "SpGEMMHashR": "spgemm_sparse_hash_row",
        "SpGEMMHashC": "spgemm_sparse_hash_col",
        "SpGEMMSortR": "spgemm_sparse_sort_row",
        "SpGEMMSortC": "spgemm_sparse_sort_col",
        "SpGEMMCusp": "spgemm_cusparse",
    }

    is_reverse = False
    # generate a_matrix.bin and b_matrix.bin
    if algorithm.startswith("reverse_"):
        algorithm = algorithm[8:]
        gen_matrix(a_matrix, config.TEMP_PATH, "b_matrix.bin")
        gen_matrix(b_matrix, config.TEMP_PATH, "a_matrix.bin")
        is_reverse = True
    else:
        gen_matrix(a_matrix, config.TEMP_PATH, "a_matrix.bin")
        gen_matrix(b_matrix, config.TEMP_PATH, "b_matrix.bin")

    try:
        output = subprocess.check_output(
            [
                f"/root/workspace/samsungstf/build/Release/algorithm/{dict_name[algorithm]}",
                f"{config.TEMP_PATH}/a_matrix.bin",
                f"{config.TEMP_PATH}/b_matrix.bin",
            ]
        )
        output = output.decode("utf-8").strip().split("\n")[-1].split()
        # 1) host_to_device_time 2) kernel_time 3) device_to_host_time 4) total_time
        output = list(map(float, output))
    except Exception as e:
        print(f"Error: {e}")
        output = [1000, 1000, 1000, 1000]
    return output
