from parameter import *
import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"
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

SAVE_DIR = "/root/workspace/samsungstf/training_results"
PREFIX = "bert_100000.pth"
THRESHOLD = 0.1
if not len(sys.argv) == 1:
    PREFIX = sys.argv[1]

if "small" in PREFIX:
    # Load pretrained tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
    config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
    layer_count = 4
else:
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    config = BertConfig.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    layer_count = 2
model = BertForMaskedLM(config)

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
# Define a dictionary to hold the input activations of each transformer layer
activations = {}


def get_sparsity(tensor):
    ## num_zeros = (tensor == 0).sum().item()
    ## num_elements = tensor.numel()
    ## return num_zeros / num_elements

    # Calculate sparsity based on threshold
    num_zeros = (tensor.abs() < THRESHOLD).sum().item()
    num_elements = tensor.numel()
    return num_zeros / num_elements


def hook_fn(module, input, output):
    if isinstance(module, nn.Linear):
        if module not in activations:
            activations[module] = []
        input_tensor = input[0].detach()
        sparsity = get_sparsity(input_tensor)
        activations[module].append(sparsity)


# Register hooks for each transformer layer
hooks = []
for name, layer in model.named_modules():
    layer.name_ = name
    if isinstance(layer, nn.Linear):
        # hook = layer.register_forward_hook(hook_fn)
        # hooks.append(hook)
        pass

input_visualization(model)
###############################################################################

# Load dataset and tokenizer
dataset = load_dataset("bookcorpus", data_dir="/root/workspace/samsungstf/data")

# randomly select a subset of the dataset with long paragraphs
dataset = dataset["train"].select(list(range(N_TESTS * 100)))
dataset = dataset.shuffle()
dataset = dataset.select(list(range(N_TESTS)))


# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch")

test_loader = DataLoader(dataset=tokenized_dataset, batch_size=1, shuffle=False)

# Load the model
saved_model = torch.load(os.path.join(SAVE_DIR, PREFIX))
model.load_state_dict(saved_model["model"])

attention_sparsity = {}
# Evaluate the model
model.eval()
total = 0
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(
            input_ids, attention_mask=attention_mask, output_attentions=True
        )
        # Add sparsity of attentions
        for i in range(len(outputs.attentions)):
            for j in range(outputs.attentions[i].shape[1]):
                if (i, j) not in attention_sparsity:
                    attention_sparsity[(i, j)] = []
                target_attention_mask = np.count_nonzero(attention_mask.cpu().numpy())
                base_attention = outputs.attentions[i][
                    :, j, :target_attention_mask, :target_attention_mask
                ]
                # Calculate sparsity based on threshold
                num_zeros = (base_attention.abs() < 0.05).sum().item()
                num_elements = base_attention.numel()
                sparsity = num_zeros / num_elements
                attention_sparsity[(i, j)].append(sparsity)
        total += 1


layer_stats = {}
for layer, sparsity_list in attention_sparsity.items():
    sparsity_array = np.array(sparsity_list)
    avg_sparsity = np.mean(sparsity_array).item()
    std_sparsity = np.std(sparsity_array).item()
    layer_name = f"attention_{layer[0]}_{layer[1]}"
    layer_stats[layer_name] = {
        "sparsity_average": avg_sparsity,
        "sparsity_std": std_sparsity,
        "sparsity_list": sparsity_list,
    }


draw_distribution()

json.dump(
    {
        "layer_stats": layer_stats,
        "epoch": saved_model["epoch"],
        "training_idx": saved_model["training_idx"],
    },
    open(os.path.join(SAVE_DIR, PREFIX.replace(".pth", "!.json")), "w"),
)

# Remove hooks
for hook in hooks:
    hook.remove()
