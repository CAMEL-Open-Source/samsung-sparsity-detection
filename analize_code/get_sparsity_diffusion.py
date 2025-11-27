from parameter import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
import torchvision
from tqdm import tqdm
import numpy as np
import os
import sys
import json
import random
import scipy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
SAVE_DIR = "/root/workspace/samsungstf/training_results"
PREFIX = "diffusion_relu_39000.pth"
THRESHOLD = 0.3
N_TESTS = 20
if not len(sys.argv) == 1:
    PREFIX = sys.argv[1]

# Initialize the model
if "stable" in PREFIX:
    model = UNet2DModel(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
        ),
        act_fn="relu" if ("relu" in PREFIX) else "silu",
    )
else:
    model = UNet2DModel(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        act_fn="relu" if ("relu" in PREFIX) else "silu",
    )

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

# Load the model
saved_model = torch.load(os.path.join(SAVE_DIR, PREFIX))
if "module" in list(saved_model["model"].keys())[0]:
    # remove the "module." prefix
    new_state_dict = {}
    for key in saved_model["model"]:
        new_state_dict[key.replace("module.", "")] = saved_model["model"][key]
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(saved_model["model"])

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Set the number of inference steps
num_inference_steps = 16

# Initialize the scheduler timesteps
noise_scheduler.set_timesteps(num_inference_steps)

# Evaluate the model
model.eval()
total = 0

for _ in tqdm(range(N_TESTS)):
    sample_images = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        for t in noise_scheduler.timesteps:
            # Predict the noise residual
            noise_pred = model(sample_images, t).sample

            # Compute the previous image
            sample_images = noise_scheduler.step(
                noise_pred, t, sample_images
            ).prev_sample

        # Clamp the images to [0, 1] if necessary
        sample_images = (sample_images / 2 + 0.5).clamp(0, 1)

        # Save the image
        # torchvision.utils.save_image(
        #     sample_images[0],
        #     f"diffusion_{total}.png",
        # )

        total += 1

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
        if isinstance(prev_layer, nn.GroupNorm):
            avg = prev_layer.bias.cpu().detach().numpy()
            norm = prev_layer.weight.cpu().detach().numpy()
            stats[layer.name_]["pred_sparsity"] = scipy.stats.norm.cdf(
                0, loc=avg, scale=norm
            ).mean()
            break

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
