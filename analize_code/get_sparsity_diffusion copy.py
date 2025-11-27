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

model_profiler = ModelProfiler(model)

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

target_layers = "Linear SiLU Conv2d GroupNorm ReLU Dropout".split()
layer_times = model_profiler.layer_times
total_time = sum(
    [time for layer, time in layer_times.items() if layer in target_layers]
)
print(
    layer_times["Conv2d"] / total_time,
    layer_times["GroupNorm"] / total_time,
    1 - (layer_times["Conv2d"] + layer_times["GroupNorm"]) / total_time,
)

model_profiler.report()
