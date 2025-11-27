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

N_TESTS = 1024
BATCH_SIZE = 1
DEVICE = "cuda:0"

tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
model = BertForMaskedLM(config)

# Set device
model = model.to(DEVICE)
model.eval()

time_map = {}
layer_times = {}


def record_start_time(module, input):
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    layer_times[module.name_] = start_event


def record_end_time(module, input, output):
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    torch.cuda.synchronize()
    start_event = layer_times[module.name_]
    time_map[module.name_] += start_event.elapsed_time(end_event)


for name, layer in model.named_modules():
    layer.name_ = name
    time_map[name] = 0
    layer_times[name] = None

    layer.register_forward_pre_hook(record_start_time)
    layer.register_forward_hook(record_end_time)

# Load dataset and tokenizer
dataset = load_dataset("bookcorpus", data_dir="/root/workspace/samsungstf/data")

# randomly select a subset of the dataset with long paragraphs
dataset = dataset["train"].select(list(range(N_TESTS * 100)))
dataset = dataset.shuffle()
dataset = dataset.select(list(range(N_TESTS)))


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch")

test_loader = DataLoader(
    dataset=tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False
)

for batch in tqdm(test_loader):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

print(json.dumps(time_map, indent=4))
