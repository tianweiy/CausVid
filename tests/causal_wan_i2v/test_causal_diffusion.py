from causvid.models.wan.wan_wrapper import CausalWanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from diffusers.utils import export_to_video
import torch

torch.set_grad_enabled(False)

model = CausalWanDiffusionWrapper().to("cuda").to(torch.float32)
encoder = WanTextEncoder().to("cuda").to(torch.float32)
vae = WanVAEWrapper().to("cuda").to(torch.float32)

model.set_module_grad(
    {
        "model": False
    }
)

model.model.i2v = True 
model.model.num_frame_per_block = 4 

text_prompts = [r"""A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."""] * 2

conditional_dict = encoder(
    text_prompts=text_prompts
)

noise = torch.randn(
    2, 21, 16, 60, 104, generator=torch.Generator().manual_seed(42), dtype=torch.float32
).to("cuda")

timetep = 999 * torch.ones([2, 21], device="cuda", dtype=torch.long)

with torch.no_grad():
    output = model(noise, conditional_dict, timetep)
    video = vae.decode_to_pixel(output)

video = (video * 0.5 + 0.5).cpu().detach().to(torch.float32)[0].permute(0, 2, 3, 1).numpy()

export_to_video(video, "causal_output_i2v.mp4", fps=8)
