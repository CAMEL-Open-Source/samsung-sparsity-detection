#!/usr/bin/env torchrun --nproc_per_node=4
# torchrun --nproc_per_node=4 /root/workspace/samsungstf/training_code/diffusion_training.py

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm
from dataclasses import dataclass
import torchvision


# Define the training configuration
@dataclass
class TrainingConfig:
    image_size = 128  # Target image resolution
    train_batch_size = 16
    eval_batch_size = 16  # Number of images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    stable_diffusion = True  # Use a stable version of the diffusion model
    use_relu = True  # Use ReLU activation in the model
    mixed_precision = (
        "fp16"  # Set to 'no' for float32, 'fp16' for automatic mixed precision
    )
    output_dir = "/root/workspace/samsungstf/training_results"  # Model name to save locally and on the HF Hub
    seed = 0


# Initialize the process group for DDP
def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


# Clean up the process group after training
def cleanup():
    dist.destroy_process_group()


# Function to check if the current process is the main one (rank 0)
def is_main_process():
    if "LOCAL_RANK" not in os.environ:
        return True
    return dist.get_rank() == 0


# Function to save the model checkpoint
def save_model_checkpoint(model, epoch, config):
    if is_main_process():
        os.makedirs(config.output_dir, exist_ok=True)
        model.module.save_pretrained(
            os.path.join(config.output_dir, f"diffusion-{epoch+1}")
        )


# Function to generate and save sample images
def save_sample_images(model, epoch, config, noise_scheduler, device, scaler=None):
    if is_main_process():
        model.eval()
        with torch.no_grad():
            # Sample random noise
            sample_images = torch.randn(
                config.eval_batch_size, 3, config.image_size, config.image_size
            ).to(device)

            # Inference loop
            for t in tqdm(
                reversed(range(noise_scheduler.num_train_timesteps)),
                desc="Generating images",
                total=noise_scheduler.num_train_timesteps,
                leave=False,
            ):
                # Create a tensor of timesteps for the batch
                timesteps = torch.full(
                    (sample_images.shape[0],), t, device=device, dtype=torch.long
                )
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    noise_pred = model(sample_images, timesteps).sample
                # Pass scalar t to noise_scheduler.step()
                sample_images = noise_scheduler.step(
                    noise_pred, t, sample_images
                ).prev_sample

            # Denormalize and save images
            sample_images = (sample_images / 2 + 0.5).clamp(0, 1)
            grid = torchvision.utils.make_grid(sample_images, nrow=4)
            if config.use_relu:
                torchvision.utils.save_image(
                    grid, os.path.join(config.output_dir, f"diffusion_relu_{epoch}.png")
                )
            else:
                torchvision.utils.save_image(
                    grid, os.path.join(config.output_dir, f"diffusion_{epoch}.png")
                )


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        # Get local rank from environment variable
        local_rank = int(os.environ["LOCAL_RANK"])
        setup()  # Initialize the process group
    else:
        local_rank = 0

    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Initialize the device for each process
    device = torch.device(f"cuda:{local_rank}")

    # Initialize the model
    if TrainingConfig.stable_diffusion:
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
            act_fn="relu" if TrainingConfig.use_relu else "silu",
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
            act_fn="relu" if TrainingConfig.use_relu else "silu",
        )

    # Move model to the device and wrap it in DistributedDataParallel
    model.to(device)
    if "LOCAL_RANK" in os.environ:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ]
    )

    # Load the dataset (using CIFAR-10 as an example)
    train_dataset = datasets.CIFAR10(
        root="tmp", train=True, download=True, transform=transform
    )

    # Use DistributedSampler to distribute data across GPUs
    if "LOCAL_RANK" in os.environ:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize the noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define learning rate scheduler
    def lr_lambda(current_step):
        if current_step < 500:
            return float(current_step) / float(max(1, 500))
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Set up mixed precision if specified
    scaler = torch.cuda.amp.GradScaler() if "fp16" == "fp16" else None

    training_idx = 0
    # Training loop
    for epoch in range(50):
        model.train()
        if "LOCAL_RANK" in os.environ:
            train_sampler.set_epoch(epoch)  # Set the sampler for each epoch
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/50", leave=False)
        for step, batch in enumerate(progress_bar):
            clean_images, _ = batch
            clean_images = clean_images.to(device)

            # Add random noise
            noise = torch.randn(clean_images.shape).to(device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=device
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict noise residual
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                noise_pred = model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backpropagation
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            progress_bar.set_postfix({"loss": loss.item()})

            training_idx += 1
            if training_idx % 100 == 0 and is_main_process():
                prefix = "diffusion_"
                # Save the model checkpoint
                if TrainingConfig.use_relu:
                    prefix += "relu_"
                if TrainingConfig.stable_diffusion:
                    prefix += "stable_"
                checkpoint_path = os.path.join(
                    TrainingConfig.output_dir, f"{prefix}{training_idx}.pth"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "batch_idx": step,
                        "loss": loss.item(),
                        "training_idx": training_idx,
                    },
                    checkpoint_path,
                )

        # # Save model checkpoints every save_model_epochs
        # if (epoch + 1) % 30 == 0 or epoch == 50 - 1:
        #     save_model_checkpoint(model, epoch, TrainingConfig)

        # Save sample images every save_image_epochs
        if (epoch + 1) % 10 == 0:
            save_sample_images(
                model, training_idx, TrainingConfig, noise_scheduler, device, scaler
            )

    if "LOCAL_RANK" in os.environ:
        # Clean up the process group after training
        cleanup()
