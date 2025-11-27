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

model_profiler = ModelProfiler(model)

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

target_layers = "Conv2d BatchNorm2d ReLU MaxPool2d AdaptiveAvgPool2d Linear".split()
layer_times = model_profiler.layer_times
total_time = sum(
    [time for layer, time in layer_times.items() if layer in target_layers]
)
print(
    layer_times["Conv2d"] / total_time,
    layer_times["BatchNorm2d"] / total_time,
    1 - (layer_times["BatchNorm2d"] + layer_times["Conv2d"]) / total_time,
)

model_profiler.report()
