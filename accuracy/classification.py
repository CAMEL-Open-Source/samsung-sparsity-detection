import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertConfig,
    BertModel,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys
import json
import random
import math
from benchmark.config_file import *
from typing import Optional, Tuple

# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class SparseAttentionConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.N_TESTS = 16384
        # self.N_TESTS = 1024
        self.N_CHOICES = 1024
        # self.N_CHOICES = 32
        self.CHOICE_REVERSE = False
        self.BATCH_SIZE = 1
        self.DEVICE = "cuda:0"

        self.THRESHOLD_K = 0.99
        self.TOP_K = 1
        self.MODE = "TOP_K"

        self.SAVE_DIR = "/root/workspace/samsungstf/training_results"
        self.PREFIX = "bert_small_450000.pth"

        self.total_density = 0
        self.total_correct = 0
        self.total_masked = 0
        self.best_correct = 0
        self.best_masked = 0


SPARSE = False

config = SparseAttentionConfig()
config.parse_args()


class CustomBertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        if SPARSE:
            if config.MODE == "THRESHOLD_K":
                n_attention_mask = attention_probs.shape[
                    2
                ] - attention_mask.nonzero().size(0)
                threshold = config.THRESHOLD_K / n_attention_mask
                mask = attention_probs < threshold
                attention_probs[mask] = 0
            elif config.MODE == "TOP_K":
                top_k = config.TOP_K
                # select top-k values across dim=-1
                top_values, _ = torch.topk(attention_probs, top_k, dim=-1)
                # create a mask for values that are in top-k
                mask = attention_probs >= top_values[:, :, :, -1].unsqueeze(-1)
                # set all values that are not in top-k to 0
                attention_probs[~mask] = 0

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# overwrite the original self-attention layer with the custom layer
import transformers.models.bert.modeling_bert as bert

bert.BertSelfAttention = CustomBertSelfAttention

tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
bert_config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
model = BertForMaskedLM(bert_config)

# Load the model
saved_model = torch.load(os.path.join(config.SAVE_DIR, config.PREFIX))
model.load_state_dict(saved_model["model"])
model = BertForMaskedLM.from_pretrained("google/bert_uncased_L-4_H-256_A-4")

# Set device
model = model.to(config.DEVICE)
model.eval()

density_list = []
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

    # layer.register_forward_pre_hook(record_start_time)
    # layer.register_forward_hook(record_end_time)

# Load dataset and tokenizer
# dataset = load_dataset("bookcorpus", data_dir="/root/workspace/samsungstf/data")
dataset = load_dataset("imdb")

# randomly select a subset of the dataset with long paragraphs
# dataset = dataset["train"].select(list(range(config.N_TESTS * 100)))
# dataset = dataset.shuffle()
# dataset = dataset.select(list(range(config.N_TESTS)))
dataset = dataset["train"].select(list(range(config.N_TESTS)))


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
tokenized_dataset.set_format("torch")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
tokenized_dataset.set_format("torch")

test_loader = DataLoader(
    dataset=tokenized_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)
text_length = 0

for batch in tqdm(test_loader):
    input_ids = batch["input_ids"].to(config.DEVICE)
    attention_mask = batch["attention_mask"].to(config.DEVICE)
    labels = batch["labels"].to(config.DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            labels=labels,
        )
        mask_positions = labels != -100
        masked_labels = labels[mask_positions]
        masked_logits = outputs.logits[mask_positions]
        nmasked_attention = attention_mask.nonzero().size(0)

        if len(masked_labels) == 0:
            continue

        text_length += nmasked_attention
        prediction = masked_logits.argmax(dim=-1)
        corrected = torch.sum(prediction == masked_labels).item()
        masked = len(masked_labels)

        attention_results = outputs.attentions
        threshold = 0.99 / nmasked_attention
        avg_density = 0
        for i, layer_attention in enumerate(attention_results):
            density = torch.sum(layer_attention > threshold).item() / (
                nmasked_attention * layer_attention.shape[1] * layer_attention.shape[2]
            )
            avg_density += density

        density_list.append(
            {
                "density": avg_density / len(attention_results),
                "text": tokenizer.decode(input_ids[0]),
                "corrected": corrected,
                "masked": masked,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

print(f"Average text length: {text_length / len(density_list)}")
density_list = sorted(density_list, key=lambda x: x["density"], reverse=True)

SPARSE = True
if not config.CHOICE_REVERSE:
    sampled_density_list = density_list[: config.N_CHOICES]
else:
    sampled_density_list = density_list[-config.N_CHOICES :]
config.best_correct = sum([data["corrected"] for data in sampled_density_list])
config.best_masked = sum([data["masked"] for data in sampled_density_list])
print(f"Accuracy: {config.best_correct / config.best_masked}")

total_correct = 0
total_masked = 0
total_density = 0
for i, data in enumerate(sampled_density_list):
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    labels = data["labels"]

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            labels=labels,
        )
        mask_positions = labels != -100
        masked_labels = labels[mask_positions]
        masked_logits = outputs.logits[mask_positions]
        nmasked_attention = attention_mask.nonzero().size(0)

        prediction = masked_logits.argmax(dim=-1)
        corrected = torch.sum(prediction == masked_labels).item()
        masked = len(masked_labels)

        total_correct += corrected
        total_masked += masked

        attention_results = outputs.attentions
        density_ = 0
        for i, layer_attention in enumerate(attention_results):
            density_ += layer_attention.nonzero().size(0) / (
                nmasked_attention * layer_attention.shape[1] * layer_attention.shape[2]
            )
        total_density += density_ / len(attention_results)

print(f"New Accuracy: {total_correct / total_masked}")
print(f"New Density: {total_density / config.N_CHOICES}")

config.total_density = total_density / config.N_CHOICES
config.total_correct = total_correct
config.total_masked = total_masked
config.save_config()
