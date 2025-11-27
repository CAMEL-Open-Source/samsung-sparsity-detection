#!/root/workspace/parallelism_sim/venv/bin/python3
import glob
import json
import plotly.graph_objs as go
import plotly.io as pio
import copy
from benchmark.collector import *


def collect_matrix_multiplication():
    result_files = glob_files(
        "/root/workspace/samsungstf/results", "alg_matrix_multiplication_.*.json"
    )
    collector = Collector()
    for file in result_files:
        result = json.load(open(file))
        collector.collect(
            "format_conversion_time",
            None,
            func=get_bypass,
            arg=result["format_conversion_time"],
        )
        collector.collect(
            "matrix_multiplication_time",
            None,
            func=get_bypass,
            arg=result["matrix_multiplication_time"],
        )
        collector.collect("SIZE", None, func=get_bypass, arg=result["SIZE"])
        collector.collect("SPARSITY", None, func=get_bypass, arg=result["SPARSITY"])
        collector.line += 1

    collector.print()


collect_matrix_multiplication()
