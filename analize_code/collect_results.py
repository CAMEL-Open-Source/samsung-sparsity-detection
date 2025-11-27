#!/usr/bin/env python
import glob
import json
import torch
import math
import scipy.stats as stats
import numpy as np
from benchmark.collector import *

RESULT_DIR = "/root/workspace/samsungstf/training_results"

RESNET_PREFIX = "resnet18"
BERT_PREFIX = "bert_small"
DIFFUSION_PREFIX = "diffusion_relu"
RESNET_TARGET_LAYER = [
    "layer3.1.conv2",
    "layer2.1.conv1",
    "layer4.0.downsample.0",
    # "layer1.1.conv1",
]
BERT_TARGET_LAYER = [
    "attention_0_2",
    "attention_2_0",
    "attention_3_1",
    # "attention_0_2",
]
DIFFUSION_TARGET_LAYER = [
    "down_blocks.3.resnets.1.conv2",
    "down_blocks.1.resnets.0.conv1",
    "down_blocks.0.resnets.1.conv2",
    # "down_blocks.0.resnets.1.conv2",
]


def get_prefix(model):
    if "resnet18" in model:
        return RESNET_PREFIX, RESNET_TARGET_LAYER
    elif "bert" in model:
        return BERT_PREFIX, BERT_TARGET_LAYER
    elif "diffusion" in model:
        return DIFFUSION_PREFIX, DIFFUSION_TARGET_LAYER


def collect_layerwise_(collector, result, layer, stats):
    collector.collect(
        "training_idx",
        None,
        func=get_bypass,
        arg=result["training_idx"],
    )
    collector.collect(
        "layer",
        None,
        func=get_bypass,
        arg=layer,
    )
    collector.collect(
        "sparsity_average",
        None,
        func=get_bypass,
        arg=stats["sparsity_average"],
    )
    collector.collect(
        "sparsity_std",
        None,
        func=get_bypass,
        arg=np.std(stats["sparsity_list"]).item(),
    )
    # p95 and p5 percentile
    collector.collect(
        "sparsity_p5",
        None,
        func=get_bypass,
        arg=np.percentile(stats["sparsity_list"], 5).item(),
    )
    collector.collect(
        "sparsity_p95",
        None,
        func=get_bypass,
        arg=np.percentile(stats["sparsity_list"], 95).item(),
    )
    collector.collect(
        "pred_sparsity",
        None,
        func=get_bypass,
        arg=stats.get("pred_sparsity", None),
    )


def collect_layerwise():
    PREFIX, TARGET_LAYER = get_prefix("resnet18")
    resnet18_files = glob_files(RESULT_DIR, PREFIX + "_\d+.json")
    target_keys = TARGET_LAYER
    # target_keys = []

    collector = Collector()
    for file in resnet18_files:
        with open(file, "r") as f:
            result = json.load(f)

            if len(target_keys) != 0:
                for layer in target_keys:
                    stats = result["layer_stats"][layer]
                    collect_layerwise_(collector, result, layer, stats)
                    collector.line += 1
            else:
                for layer, stats in result["layer_stats"].items():
                    collect_layerwise_(collector, result, layer, stats)
                    collector.line += 1

    collector.print_2ry("layer", "training_idx")


def collect_matmul_history():
    PREFIX, TARGET_LAYER = get_prefix("resnet18")
    for layer in TARGET_LAYER:
        resnet18_files = glob_files(RESULT_DIR, PREFIX + "_\d+.json")
        weight_sparsity = np.loadtxt(
            "/root/workspace/samsungstf/results/weight_sparsity.txt"
        )
        target_index = hash(layer) % weight_sparsity.shape[1]
        weight_sparsity = weight_sparsity[:, target_index]
        results = []
        for file in resnet18_files:
            with open(file, "r") as f:
                result = json.load(f)
                if "bert" in PREFIX:
                    stats = result["layer_stats"][layer]
                else:
                    stats = result["layer_stats"][layer]
                sparsity_list = stats["sparsity_list"]
                input_sparsity_std = np.std(sparsity_list).item()
                for i, sparsity in enumerate(sparsity_list):
                    idx = (
                        len(weight_sparsity)
                        * len(results)
                        // (len(sparsity_list) * len(resnet18_files))
                    )
                    results.append(
                        {
                            "training_idx": result["training_idx"],
                            "input_sparsity": sparsity,
                            "input_sparsity_avg": stats["sparsity_average"],
                            "input_sparsity_std": input_sparsity_std,
                            "weight_sparsity": weight_sparsity[idx] * 0.01,
                            "weight_sparsity_avg": weight_sparsity[idx] * 0.01,
                            "weight_sparsity_std": 0,
                        }
                    )
        # dump results to json
        with open(
            f"/root/workspace/samsungstf/results/matmul_history_{PREFIX}_{layer}.json",
            "w",
        ) as f:
            json.dump(results, f, indent=2)
        with open(
            f"/root/workspace/samsungstf/results/matmul_history_{PREFIX}_{layer}.txt",
            "w",
        ) as f:
            for idx, result in enumerate(results):
                f.write(f"{idx / len(results)} {result['input_sparsity']}\n")


def collect_optimal_operator_gain():
    results = glob_files(
        "/root/workspace/samsungstf/results", "optimal_operator_gain_1024_.*.json"
    )
    collector = Collector()
    for result in results:
        with open(result, "r") as f:
            result = json.load(f)

            if result["matmul_time"][1] > 500:
                result["matmul_time"][1] = None

            collector.collect(
                "algorithm",
                None,
                func=get_bypass,
                arg=result["ALGORITHM"],
            )
            collector.collect(
                "sparsity",
                None,
                func=get_bypass,
                arg=result["A_SPARSITY"],
            )
            collector.collect(
                "matmul_time",
                None,
                func=get_bypass,
                arg=result["matmul_time"][1],
            )

            collector.line += 1

    collector.sort(
        "algorithm",
        reference_list="GEMM SpMMRowR SpMMRowC SpMMElemR SpMMElemC SpGEMMHashR SpGEMMHashC SpGEMMSortR SpGEMMSortC",
    )

    collector.print_2ry("algorithm", "sparsity")


def collect_optimal_operator_profile():
    results = glob_files(
        "/root/workspace/samsungstf/results", "optimal_operator_profile_1024_.*.json"
    )
    collector = Collector()
    for result in results:
        with open(result, "r") as f:
            result = json.load(f)

            collector.collect(
                "algorithm",
                None,
                func=get_bypass,
                arg=result["ALGORITHM"],
            )
            collector.collect(
                "a_sparsity",
                None,
                func=get_bypass,
                arg=result["A_SPARSITY"],
            )
            collector.collect(
                "b_sparsity",
                None,
                func=get_bypass,
                arg=result["B_SPARSITY"],
            )
            collector.collect(
                "matmul_time",
                None,
                func=get_bypass,
                arg=result["matmul_time"][1],
            )

            collector.line += 1

    collector.print()


def collect_variance_sensitivity():
    results = glob_files(
        "/root/workspace/samsungstf/results",
        "variance_sensitivity_1024_.*_0\.7_.*.json",
    )
    collector = Collector()
    for result in results:
        with open(result, "r") as f:
            result = json.load(f)

            collector.collect(
                "algorithm",
                None,
                func=get_bypass,
                arg=result["ALGORITHM"],
            )
            collector.collect(
                "a_sparsity",
                None,
                func=get_bypass,
                arg=result["A_SPARSITY"],
            )
            collector.collect(
                "b_sparsity",
                None,
                func=get_bypass,
                arg=result["B_SPARSITY"],
            )
            collector.collect(
                "a_sparsity_std",
                None,
                func=get_bypass,
                arg=result["A_SPARSITY_STD"],
            )
            collector.collect(
                "matmul_time_avg",
                None,
                func=get_bypass,
                arg=result["matmul_time_avg"][1],
            )
            collector.collect(
                "matmul_time_std",
                None,
                func=get_bypass,
                arg=result["matmul_time_std"][1],
            )
            collector.line += 1

    collector.sort(
        "algorithm",
        reference_list="GEMM SpMMRowR SpMMRowC SpMMElemR SpMMElemC SpGEMMHashR SpGEMMHashC SpGEMMSortR SpGEMMSortC",
    )
    collector.sort("a_sparsity_std", reference_list=[0, 0.04, 0.08, 0.12, 0.16, 0.2])

    collector.print_2ry("algorithm", "a_sparsity_std")


def collect_period_sparsity_detection_sensitivity():
    results = glob_files(
        "/root/workspace/samsungstf/results",
        "period_sparsity_detection_100sensitivity_.*\.json",
    )
    dict_map = {}

    collector = Collector()
    for result in results:
        with open(result, "r") as f:
            result = json.load(f)

            layer_model = Path(result["SPARSITY_LIST"]).stem.replace(
                "matmul_history_", ""
            )
            key_name = layer_model.split("_")[0] + "_" + str(result["DETECT_PERIOD"])
            collector.collect(
                "target_layer",
                None,
                func=get_bypass,
                arg=layer_model,
            )
            collector.collect(
                "detect_period",
                None,
                func=get_bypass,
                arg=result["DETECT_PERIOD"],
            )
            collector.collect(
                "matmul_time",
                None,
                func=get_bypass,
                arg=result["total_matmul_time"],
            )
            collector.collect(
                "detection_time",
                None,
                func=get_bypass,
                arg=result["total_detection_time"],
            )
            collector.collect(
                "matmul_detection_time",
                None,
                func=get_bypass,
                arg=result["total_matmul_time"] + result["total_detection_time"],
            )
            collector.collect(
                "format_conversion_time",
                None,
                func=get_bypass,
                arg=result["total_format_conversion_time"],
            )
            collector.collect(
                "transpose_time",
                None,
                func=get_bypass,
                arg=result["total_transpose_time"],
            )

            if key_name not in dict_map:
                dict_map[key_name] = {}
                dict_map[key_name]["matmul_time"] = result["total_matmul_time"]
                dict_map[key_name]["detection_time"] = result["total_detection_time"]
                dict_map[key_name]["matmul_detection_time"] = (
                    result["total_matmul_time"] + result["total_detection_time"]
                )
                dict_map[key_name]["format_conversion_time"] = result[
                    "total_format_conversion_time"
                ]
                dict_map[key_name]["transpose_time"] = result["total_transpose_time"]
                dict_map[key_name]["total_count"] = 1
            else:
                dict_map[key_name]["matmul_time"] += result["total_matmul_time"]
                dict_map[key_name]["detection_time"] += result["total_detection_time"]
                dict_map[key_name]["matmul_detection_time"] += (
                    result["total_matmul_time"] + result["total_detection_time"]
                )
                dict_map[key_name]["format_conversion_time"] += result[
                    "total_format_conversion_time"
                ]
                dict_map[key_name]["transpose_time"] += result["total_transpose_time"]
                dict_map[key_name]["total_count"] += 1
            collector.line += 1

    collector.print_2ry("target_layer", "detect_period")

    collector = Collector()
    for model, result in dict_map.items():
        model, detect_period = model.split("_")
        collector.collect(
            "model",
            None,
            func=get_bypass,
            arg=model,
        )
        collector.collect(
            "detect_period",
            None,
            func=get_bypass,
            arg=detect_period,
        )
        for key, value in result.items():
            if key == "total_count":
                continue
            collector.collect(
                key,
                None,
                func=get_bypass,
                arg=value / result["total_count"],
            )
        collector.line += 1

    collector.sort("model", reference_list="resnet18 bert diffusion")
    collector.print_2ry("model", "detect_period")


def table_collect_period_sparsity_detection_sensitivity():
    results = glob_files(
        "/root/workspace/samsungstf/results",
        "period_sparsity_detection_sensitivity_.*\.json",
    )
    layer_target = "diffusion_relu_down_blocks.0.resnets.1.conv2"

    table = Table()
    for result in results:
        with open(result, "r") as f:
            result = json.load(f)

            layer_model = Path(result["SPARSITY_LIST"]).stem.replace(
                "matmul_history_", ""
            )
            if layer_model != layer_target:
                continue

            if result["DETECT_PERIOD"] >= 2:
                continue

            table.collect(
                "detect_period",
                None,
                func=get_bypass_all,
                arg=(
                    result["DETECT_PERIOD"],
                    [hist["matmul_time"][1] for hist in result["method_history"]],
                ),
            )

    table.print()


def collect_period_sparsity_detection_method():
    results = glob_files(
        "/root/workspace/samsungstf/results",
        "period_sparsity_detection_method_.*\.json",
    )
    dict_map = {}

    collector = Collector()
    for result in results:
        with open(result, "r") as f:
            result = json.load(f)

            layer_model = Path(result["SPARSITY_LIST"]).stem.replace(
                "matmul_history_", ""
            )
            key_name = (layer_model.split("_")[0], result["DETECT_PRESET"])
            collector.collect(
                "target_layer",
                None,
                func=get_bypass,
                arg=layer_model,
            )
            collector.collect(
                "detect_preset",
                None,
                func=get_bypass,
                arg=result["DETECT_PRESET"],
            )
            collector.collect(
                "matmul_time",
                None,
                func=get_bypass,
                arg=result["total_matmul_time"],
            )
            collector.collect(
                "detection_time",
                None,
                func=get_bypass,
                arg=result["total_detection_time"],
            )
            collector.collect(
                "matmul_detection_time",
                None,
                func=get_bypass,
                arg=result["total_matmul_time"] + result["total_detection_time"],
            )
            collector.collect(
                "format_conversion_time",
                None,
                func=get_bypass,
                arg=result["total_format_conversion_time"],
            )
            collector.collect(
                "transpose_time",
                None,
                func=get_bypass,
                arg=result["total_transpose_time"],
            )

            if key_name not in dict_map:
                dict_map[key_name] = {}
                dict_map[key_name]["matmul_time"] = result["total_matmul_time"]
                dict_map[key_name]["detection_time"] = result["total_detection_time"]
                dict_map[key_name]["matmul_detection_time"] = (
                    result["total_matmul_time"] + result["total_detection_time"]
                )
                dict_map[key_name]["format_conversion_time"] = result[
                    "total_format_conversion_time"
                ]
                dict_map[key_name]["transpose_time"] = result["total_transpose_time"]
                dict_map[key_name]["total_count"] = 1
            else:
                dict_map[key_name]["matmul_time"] += result["total_matmul_time"]
                dict_map[key_name]["detection_time"] += result["total_detection_time"]
                dict_map[key_name]["matmul_detection_time"] += (
                    result["total_matmul_time"] + result["total_detection_time"]
                )
                dict_map[key_name]["format_conversion_time"] += result[
                    "total_format_conversion_time"
                ]
                dict_map[key_name]["transpose_time"] += result["total_transpose_time"]
                dict_map[key_name]["total_count"] += 1
            collector.line += 1

    collector.print_2ry("target_layer", "detect_preset")

    collector = Collector()

    for model, result in dict_map.items():
        model, detect_preset = model
        collector.collect(
            "model",
            None,
            func=get_bypass,
            arg=model,
        )
        collector.collect(
            "detect_preset",
            None,
            func=get_bypass,
            arg=detect_preset,
        )
        for key, value in result.items():
            if key == "total_count":
                continue
            collector.collect(
                key,
                None,
                func=get_bypass,
                arg=value / result["total_count"],
            )
        collector.line += 1

    collector.sort(
        "detect_preset", reference_list="Fixed Accurate Period Hyper-Predict"
    )
    collector.sort("model", reference_list="resnet18 bert diffusion")
    collector.print()
    # collector.print_2ry("model", "detect_preset")


def collect_period_detection():
    FIXED_FILE = "/root/workspace/samsungstf/cnn_fixed.json"
    OPT_FILE = FIXED_FILE.replace("fixed", "opt")
    fixed = json.load(open(FIXED_FILE, "r"))
    opt = json.load(open(OPT_FILE, "r"))
    for i in range(len(fixed["method_history"])):
        fixed_history = fixed["method_history"][i]
        opt_history = opt["method_history"][i]
        print(opt_history["matmul_time"][1] / fixed_history["matmul_time"][1])


def collect_period_detection_():
    FIXED_FILE = "/root/workspace/samsungstf/config.json"
    fixed = json.load(open(FIXED_FILE, "r"))
    for i in range(len(fixed["method_history"])):
        fixed_history = fixed["method_history"][i]
        print(fixed_history["matmul_time"][1])


def table_collect_optimal_gain():
    results = glob_files(
        "/root/workspace/samsungstf/",
        ".*\.json",
        (
            0,
            "GEMM SpMMRowR SpMMRowC SpMMElemR SpMMElemC SpGEMMHashR SpGEMMHashC SpGEMMSortR SpGEMMSortC".split(),
        ),
    )

    table = Table()
    for file in results:
        with open(file, "r") as f:
            result = json.load(f)

            table.collect(
                "detect_period",
                None,
                func=get_bypass_all,
                arg=(
                    file,
                    [hist["matmul_time"][1] for hist in result["method_history"]],
                ),
            )

    table.print()


collect_optimal_operator_profile()
