import os
import torch
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    BertConfig,
)
from datasets import load_dataset
import transformers
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

USE_BERT_SMALL = False

# Load dataset
dataset = load_dataset("bookcorpus", data_dir="/root/workspace/samsungstf/data")

if USE_BERT_SMALL:
    # Load pretrained tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
    config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
else:
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    config = BertConfig.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

model = BertForMaskedLM(config)


# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=128)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]


# Custom callback for logging and saving checkpoints
class CustomCallback(TrainerCallback):
    def __init__(self, save_steps, output_dir):
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.training_idx = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.training_idx += 1
        if self.training_idx % self.save_steps == 0:
            prefix = "bert_"
            if USE_BERT_SMALL:
                prefix += "small_"
            checkpoint_path = os.path.join(
                self.output_dir,
                f"{prefix}{self.training_idx}.pth",
            )
            torch.save(
                {
                    "model": kwargs["model"].state_dict(),
                    "epoch": state.epoch,
                    "step": state.global_step,
                    "training_idx": self.training_idx,
                },
                checkpoint_path,
            )
        return control


# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=200,
)

# Define data collator
data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[
        CustomCallback(
            save_steps=1000, output_dir="/root/workspace/samsungstf/training_results"
        )
    ],
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(f"Perplexity: {torch.exp(torch.tensor(results['eval_loss']))}")
