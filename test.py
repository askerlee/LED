import argparse
import imageio
from led.pipelines.led_pipeline import LEDPipeline
from PIL import Image
import numpy as np
import os
import torch

all_latents = []
all_timesteps = []
def collect_latents(t, latents):
    global all_latents, all_timesteps
    all_latents.append(latents)
    all_timesteps.append(t.reshape(1))
    return {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--outfiles", type=str, nargs="+", default=[])
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()
                        
args = parse_args()

# Passing an extra cached_folder allows loading the pretrained weights when the script is not in the same directory
# as the pretrained weights. LEDPipeline has its own weight loading mechanism, and the first 'pretrained_weights'
# argument won't be passed to LEDPipeline.__init__() as an argument by DiffusionPipeline.from_pretrained().
# Therefore, we have to repeat the 'pretrained_weights' argument in an extra cached_folder argument.
led = LEDPipeline.from_pretrained('pretrained_weights', cached_folder='pretrained_weights')
led = led.to("cuda")

all_input_files = []
for input in args.input:
    if os.path.isfile(input):
        input_files = [input]
    else:
        input_files = os.listdir(input)
        input_files = [ os.path.join(input, input_file) for input_file in input_files]
        input_files = sorted(input_files)

    all_input_files.extend(input_files)

all_output_files = []
if args.outdir:
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    all_output_files = [os.path.join(args.outdir, os.path.basename(input_file)) for input_file in all_input_files]
else:
    all_output_files = args.outfiles
    assert len(all_output_files) == len(all_input_files), "Number of input files and output files must be the same."

for i_file, input_file in enumerate(all_input_files):
    print(input_file)
    input = Image.open(input_file)
    input = input.convert("RGB")
    # Pad input to be width == height
    width, height = input.size
    new_width, new_height = max(width, height), max(width, height)
    new_input = Image.new("RGB", (new_width, new_height))
    if width != height:    
        if width > height:
            pad = (0, (width - height) // 2)
        elif height > width:
            pad = ((height - width) // 2, 0)
    else:
        pad = (0, 0)

    # pad indicates the upper left corner to paste the input.
    new_input.paste(input, pad)
    new_input = new_input.resize((512, 512))
    print(f"{width}x{height} -> {new_width}x{new_height} -> {new_input.size}")

    with torch.no_grad():
        # num_cond_steps: 800.
        # new_input: cond_image, one or list of low-quality images of interest.
        led_enhancement = led(cond_image=new_input, callback_on_step_end=collect_latents)[0]

    # all_latents: 41 tensors of [1, 3, 512, 512].
    # all_timesteps: 41 tensors of [800, 780, 760, ..., 20, 0]

    # [512, 512, 3] -> [600, 900, 3]
    led_enhancement_pil = Image.fromarray(led_enhancement)
    led_enhancement_pil = led_enhancement_pil.resize((new_width, new_height))
    led_enhancement = np.array(led_enhancement_pil)
    # pad: (width_pad, height_pad)
    led_enhancement = led_enhancement[pad[1]:pad[1]+height, pad[0]:pad[0]+width]
    out_path = all_output_files[i_file]
    imageio.imwrite(out_path, led_enhancement)
    print(f"Saved image to {out_path}")
