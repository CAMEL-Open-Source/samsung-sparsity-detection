import torch
import time
from torch import nn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Load the pipeline
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
eslaped_time = {}


def pre_conv_hook(module, input):
    module.start_time = time.time()


def post_conv_hook(module, input, output):
    time_spent = time.time() - module.start_time
    if hasattr(module, "total_time"):
        module.total_time += time_spent
    else:
        module.total_time = time_spent


for module in pipe.unet.modules():
    module.register_forward_pre_hook(pre_conv_hook)
    module.register_forward_hook(post_conv_hook)

for module in pipe.vae.modules():
    module.register_forward_pre_hook(pre_conv_hook)
    module.register_forward_hook(post_conv_hook)

for module in pipe.text_encoder.modules():
    module.register_forward_pre_hook(pre_conv_hook)
    module.register_forward_hook(post_conv_hook)


# Measure total execution time
start_time = time.time()
image = pipe(prompt).images[0]
total_time = time.time() - start_time

for module in pipe.unet.modules():
    if hasattr(module, "total_time"):
        eslaped_time[module.__class__.__name__] = (
            eslaped_time.get(module.__class__.__name__, 0) + module.total_time
        )

for module in pipe.vae.modules():
    if hasattr(module, "total_time"):
        eslaped_time[module.__class__.__name__] = (
            eslaped_time.get(module.__class__.__name__, 0) + module.total_time
        )

for module in pipe.text_encoder.modules():
    if hasattr(module, "total_time"):
        eslaped_time[module.__class__.__name__] = (
            eslaped_time.get(module.__class__.__name__, 0) + module.total_time
        )

print("Total execution time: ", total_time)
print(eslaped_time)

# Save the image
image.save("astronaut_rides_horse.png")


def collect_params(module, params):
    for p in module.parameters():
        params[module.__class__.__name__] = (
            params.get(module.__class__.__name__, 0) + p.numel()
        )


total_params = 0
params = {}
for module in pipe.unet.modules():
    collect_params(module, params)
total_params += sum(pipe.unet.numel() for pipe.unet in pipe.unet.parameters())

for module in pipe.vae.modules():
    collect_params(module, params)
total_params += sum(pipe.vae.numel() for pipe.vae in pipe.vae.parameters())

for module in pipe.text_encoder.modules():
    collect_params(module, params)
total_params += sum(
    pipe.text_encoder.numel() for pipe.text_encoder in pipe.text_encoder.parameters()
)

print("Total parameters: ", total_params)
print(params)
