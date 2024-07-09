import imageio
from PIL import Image
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--op", type=str, choices=["repeat-channel"], required=True)
    # The green channel contains the most information.
    parser.add_argument("--channel-idx", type=int, default=1)
    return parser.parse_args()

args = parse_args()
if args.op == "repeat-channel":
    img = Image.open(args.input)
    img = img.convert("RGB")
    img_np = np.array(img)
    img_np = np.repeat(img_np[:, :, [args.channel_idx]], 3, axis=2)
    img = Image.fromarray(img_np)
    img.save(args.output)
    print(f"Saved image to {args.output}")

