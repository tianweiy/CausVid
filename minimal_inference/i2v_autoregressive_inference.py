from causvid.models.wan.causal_inference_i2v import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextImagePairDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os
from torchvision import transforms
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--data_path", type=str)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'model']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

transform = transforms.Compose([
    transforms.Resize((480, 832)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = TextImagePairDataset(args.data_path, transform=transform)

sampled_noise = torch.randn(
    [1, 20, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    data = dataset[prompt_index]
    prompts = [data['caption']]
    image = data['image'].unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)
    # Encode the image with VAE into initial latent
    initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        initial_latent=initial_latent
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
