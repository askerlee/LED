from typing import Union, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from torchvision import transforms as tvt


def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents

all_latents = []
all_timesteps = []
def collect_latents(pipe, i, t, var_dict):
    global all_latents, all_timesteps
    all_latents.append(var_dict['latents'])
    all_timesteps.append(t.reshape(1))
    return {}

@torch.no_grad()
def ddim_inversion(imgname: str, orig_prompt: str, verify_prompt: str, 
                   num_steps: int = 50, inv_start_step: int = 10, 
                   do_edit: Optional[bool] = False) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    inverse_scheduler = DDIMInverseScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                   scheduler=inverse_scheduler,
                                                   safety_checker=None,
                                                   torch_dtype=dtype)
    pipe.to(device)

    input_img = load_image(imgname, target_size=(512, 512)).to(device=device, dtype=dtype)
    latents = img_to_latents(input_img, pipe.vae)

    global all_latents, all_timesteps
    final_inv_latents, _ = pipe(prompt=orig_prompt, negative_prompt="", guidance_scale=4.,
                                width=input_img.shape[-1], height=input_img.shape[-2],
                                output_type='latent', return_dict=False,
                                num_inference_steps=num_steps, latents=latents, 
                                callback_on_step_end=collect_latents,
                                callback_on_step_end_tensor_inputs=['latents'])

    # all_latents: 30 tensors of shape [1, 4, 64, 64].
    # final_inv_latents == all_latents[-1]
    inv_latents = all_latents[-inv_start_step]
    all_timesteps = torch.cat(all_timesteps, dim=0)
    all_timesteps = all_timesteps[:-inv_start_step].flip([0])
    # num_inv_steps: 40.
    num_inv_steps = num_steps - inv_start_step

    if do_edit:
        pipe.scheduler = DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler')
        # The following statement doesn't work: DDIMScheduler will reinitialize its timesteps
        # through retrieve_timesteps() in pipe.__call__().
        # pipe.scheduler.timesteps = all_timesteps
        # So we need to set num_train_timesteps and num_inference_steps to the correct value, 
        # so that the scheduler can recreate identical timesteps as all_timesteps.
        # num_train_timesteps: 1000 * 40 // 50 = 800.
        pipe.scheduler.config.num_train_timesteps = 1000 * num_inv_steps // num_steps
        image = pipe(prompt=verify_prompt, negative_prompt="", guidance_scale=4.,
                     num_inference_steps=num_inv_steps, latents=inv_latents)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(tvt.ToPILImage()(input_img[0]))
        ax[1].imshow(image.images[0])
        plt.show()

    return inv_latents

if __name__ == '__main__':
    ddim_inversion('./dog-runs-grass.jpg', orig_prompt="dog runs on grass", 
                   verify_prompt="cat runs on grass", num_steps=50, 
                   inv_start_step=10, do_edit=True)
    