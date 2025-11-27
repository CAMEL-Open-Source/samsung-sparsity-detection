#!/usr/bin/env python
import math
import json
import glob
from benchmark.run_shell import *


def matrix_multiplication():
    cmd = f"python /root/workspace/samsungstf/algorithm/matrix_multiplication.py --SIZE={{0}} --OUTPUT_FILE_PATH=/root/workspace/samsungstf/results/alg_matrix_multiplication_{{0}}.json"
    mat_size = "64 128 256 512 1024 2048 4096".split()

    cmd_list = []
    temp_list = format_matrix_cmd(cmd, mat_size)
    cmd_list += temp_list

    run_benchmark(cmd_list, 1, False)


matrix_multiplication()
