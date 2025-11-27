from parameter import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys
import json
import random
import scipy

SAVE_DIR = "/root/workspace/samsungstf/training_results"
PREFIX = "resnet18_5300.pth"
if not len(sys.argv) == 1:
    PREFIX = sys.argv[1]


# Define the ResNet18 model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1000)  # ImageNet has 1000 classes

# Set device
if torch.cuda.is_available():
    # random select a GPU
    num_gpus = torch.cuda.device_count()
    selected_gpu = random.randint(0, num_gpus - 1)
    device = torch.device(f"cuda:{selected_gpu}")
else:
    device = torch.device("cpu")
model = model.to(device)


###############################################################################
# Define a dictionary to hold the input activations of each convolutional layer
activations = {}


def get_sparsity(tensor):
    num_zeros = (tensor == 0).sum().item()
    num_elements = tensor.numel()
    return num_zeros / num_elements


def hook_fn(module, input, output):
    if isinstance(module, nn.Conv2d):
        if module not in activations:
            activations[module] = []
        input_tensor = input[0].detach()
        sparsity = get_sparsity(input_tensor)
        activations[module].append(sparsity)


# Register hooks for each convolutional layer
hooks = []
for name, layer in model.named_modules():
    layer.name_ = name
    if isinstance(layer, nn.Conv2d):
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

input_visualization(model)
###############################################################################

# Data loaders
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_dataset = datasets.ImageFolder(
    root="/root/dataset/ImageNet/imagenet_val",
    transform=transform,
)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# Load the model
saved_model = torch.load(os.path.join(SAVE_DIR, PREFIX))
model.load_state_dict(saved_model["model"])

# Evaluate the model
model.eval()
total = 0
with torch.no_grad():
    for x, y in tqdm(test_loader):
        x = x.to(device=device)
        y = y.to(device=device)
        scores = model(x)
        _, predictions = scores.max(1)
        total += 1
        if total == N_TESTS:
            break

# Calculate average and variance of input sparsity for each convolutional layer
stats = {}
idx = 0
for layer, sparsity_list in activations.items():
    idx += 1
    sparsity_array = np.array(sparsity_list)
    avg_sparsity = np.mean(sparsity_array)
    var_sparsity = np.var(sparsity_array)
    stats[layer.name_] = {
        "sparsity_average": avg_sparsity,
        "sparsity_variance": var_sparsity,
        "sparsity_list": sparsity_list,
    }
    # get L1 norm of the weights
    stats[layer.name_]["l1_norm"] = torch.norm(layer.weight, 1).item()
    # get batch norm statistics
    # find the prev BatchNorm layer (assuming model structure consistency)
    prev_layers = list(model.modules())[: list(model.modules()).index(layer)]
    for prev_layer in reversed(prev_layers):
        if isinstance(prev_layer, nn.BatchNorm2d):
            avg = prev_layer.bias.cpu().detach().numpy()
            norm = prev_layer.weight.cpu().detach().numpy()
            if True:
                stats[layer.name_]["pred_sparsity"] = scipy.stats.norm.cdf(
                    0, loc=avg, scale=norm
                ).mean()
            break

"""
module_dict = {name: module for name, module in model.named_modules()}
stats["layer1.0.conv1"]["pred_sparsity"] = np.power(
    scipy.stats.norm.cdf(
        0,
        module_dict["bn1"].bias.cpu().detach().numpy(),
        module_dict["bn1"].weight.cpu().detach().numpy(),
    ),
    2,
).mean()

stats["layer1.1.conv1"]["pred_sparsity"] = np.power(
    scipy.stats.norm.cdf(
        0,
        module_dict["bn1"].bias.cpu().detach().numpy(),
        module_dict["bn1"].weight.cpu().detach().numpy(),
    ),
    2,
).mean()
"""

draw_distribution()

json.dump(
    {
        "layer_stats": stats,
        "epoch": saved_model["epoch"],
        "batch_idx": saved_model["batch_idx"],
        "training_idx": saved_model["training_idx"],
        "loss": saved_model["loss"],
    },
    open(os.path.join(SAVE_DIR, PREFIX.replace(".pth", ".json")), "w"),
)

# Remove hooks
for hook in hooks:
    hook.remove()
