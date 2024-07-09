import argparse
import imageio
from led.pipelines.led_pipeline import LEDPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()
                        
args = parse_args()

led = LEDPipeline.from_pretrained('pretrained_weights')
# led.to("cuda") doesn't work.
#led.cuda()
led = led.to("cuda")
led_enhancement = led(args.input)[0]
imageio.imwrite(args.output, led_enhancement)
print(f"Saved image to {args.output}")
