import torch
from diffusers import FlowMatchEulerDiscreteScheduler

scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(10)

x_noise = torch.ones(1, 1) * 3.0
x_data = torch.zeros(1, 1)

print("=== v = data - noise (What your model learned) ===")
sample = x_noise.clone()
for t in scheduler.timesteps:
    sigma = scheduler.sigmas[scheduler.step_index]
    # True sample would be (1-sigma)*x_data + sigma*x_noise
    # And v would be x_data - x_noise
    v = x_data - x_noise
    sample = scheduler.step(model_output=v, timestep=t, sample=sample).prev_sample
    print(f"Step {scheduler.step_index}, sample: {sample.item()}")

print("\n=== v = noise - data (What diffusers expects) ===")
scheduler._step_index = None # Reset step index
sample = x_noise.clone()
for t in scheduler.timesteps:
    sigma = scheduler.sigmas[scheduler.step_index]
    v = x_noise - x_data
    sample = scheduler.step(model_output=v, timestep=t, sample=sample).prev_sample
    print(f"Step {scheduler.step_index}, sample: {sample.item()}")
