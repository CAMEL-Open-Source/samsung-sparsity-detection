import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

mpl.use("Agg")

DIST_VISUALIZATION = False
N_TESTS = 1000

input_visualization_history = {}
output_visualization_history = {}
layer_map = {}


def hook_input_visualization(module, input, output):
    global input_visualization_history
    global output_visualization_history
    try:
        input_1d = input[0].flatten()
        output_1d = output.flatten()
    except:
        return
    if module.name_ not in input_visualization_history:
        input_visualization_history[module.name_] = input_1d.detach().cpu().numpy()
        output_visualization_history[module.name_] = output_1d.detach().cpu().numpy()
    else:
        input_visualization_history[module.name_] = np.concatenate(
            (input_visualization_history[module.name_], input_1d.detach().cpu().numpy())
        )
        output_visualization_history[module.name_] = np.concatenate(
            (
                output_visualization_history[module.name_],
                output_1d.detach().cpu().numpy(),
            )
        )


def input_visualization(model):
    if not DIST_VISUALIZATION:
        return
    for name, layer in model.named_modules():
        hook = layer.register_forward_hook(hook_input_visualization)
        layer_map[name] = layer


class ModelProfiler:
    def __init__(self, model):
        self.model = model
        self.layer_times = {}
        self.start_times = {}

        # Register hooks for all submodules
        for name, layer in model.named_modules():
            layer.register_forward_pre_hook(self._pre_hook_fn(layer._get_name()))
            layer.register_forward_hook(self._hook_fn(layer._get_name()))

    def _pre_hook_fn(self, name):
        def pre_hook(layer, input):
            self.start_times[name] = time.time()

        return pre_hook

    def _hook_fn(self, name):
        def hook(layer, input, output):
            end_time = time.time()
            exec_time = end_time - self.start_times[name]
            if name not in self.layer_times:
                self.layer_times[name] = 0
            self.layer_times[name] += exec_time

        return hook

    def report(self):
        for layer_name, exec_time in self.layer_times.items():
            print(f"Layer: {layer_name}, Time: {exec_time:.6f} seconds")


def draw_distribution():
    if not DIST_VISUALIZATION:
        return
    idx = 0
    for name, input in input_visualization_history.items():
        filtered_input = input[input != 0]
        sparsity = 1 - (len(filtered_input) / len(input))
        avg = np.mean(input)
        std = np.std(input)
        plt.hist(filtered_input, bins=1000, density=True)
        plt.title(f"Sparsity: {sparsity:.2f}, avg: {avg:.2f}, std: {std:.2f}")
        plt.savefig(f"distribution_{idx:03d}_input_{name}.png")
        plt.close()
        idx += 1

    last_layer_norm = None
    idx = 0
    for name, input in output_visualization_history.items():
        filtered_input = input[input != 0]
        sparsity = 1 - (len(filtered_input) / len(input))
        avg = np.mean(input)
        std = np.std(input)
        plt.hist(filtered_input, bins=1000, density=True)
        plt.title(f"Sparsity: {sparsity:.2f}, avg: {avg:.2f}, std: {std:.2f}")
        plt.savefig(f"distribution_{idx:03d}_output_{name}.png")
        plt.close()

        if "LayerNorm" in name:
            last_layer_norm = name
            avg = np.mean(layer_map[name].bias.cpu().detach().numpy())
            std = np.mean(layer_map[name].weight.cpu().detach().numpy())
            random_values = np.random.normal(avg, std, 10000)
            plt.hist(random_values, bins=1000, density=True)
            plt.title(f"avg: {avg:.2f}, std: {std:.2f}")
            plt.savefig(f"distribution_{idx:03d}_predout_{name}.png")
            plt.close()
        if "attention.self.query" in name or "attention.self.key" in name:
            avg_input = np.mean(layer_map[last_layer_norm].bias.cpu().detach().numpy())
            std_input = np.mean(
                layer_map[last_layer_norm].weight.cpu().detach().numpy()
            )
            avg_weight = np.mean(layer_map[name].weight.cpu().detach().numpy())
            std_weight = np.std(layer_map[name].weight.cpu().detach().numpy())
            shape = layer_map[name].weight.shape
            # avg = (avg_input / (std_input**2) + avg_weight / (std_weight**2)) / (
            #     1 / (std_input**2) + 1 / (std_weight**2)
            # )
            # std = np.sqrt(1 / (1 / (std_input**2) + 1 / (std_weight**2)))
            # avg *= shape[1]
            # std *= shape[1]
            avg = shape[1] * avg_input * avg_weight
            std = np.sqrt(
                shape[1]
                * (
                    (std_input**2) * (std_weight**2)
                    + (avg_input**2) * (std_weight**2)
                    + (avg_weight**2) * (std_input**2)
                )
            )
            random_values = np.random.normal(avg, std, 10000)
            plt.hist(random_values, bins=1000, density=True)
            plt.title(f"avg: {avg:.2f}, std: {std:.2f}")
            plt.savefig(f"distribution_{idx:03d}_predout_{name}.png")
            plt.close()

        idx += 1
