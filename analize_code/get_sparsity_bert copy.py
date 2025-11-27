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
PREFIX = "bert_454000.pth"
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

model_profiler = ModelProfiler(model)

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

target_layers = "LayerNorm Dropout GELUActivation Linear".split()
layer_times = model_profiler.layer_times
total_time = sum(
    [time for layer, time in layer_times.items() if layer in target_layers]
)
print(
    layer_times["Linear"] / total_time,
    layer_times["LayerNorm"] / total_time,
    1 - (layer_times["Linear"] + layer_times["LayerNorm"]) / total_time,
)

model_profiler.report()
