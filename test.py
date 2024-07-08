
import imageio
from led.pipelines.led_pipeline import LEDPipeline
led = LEDPipeline.from_pretrained('pretrained_weights')
# led.to("cuda") doesn't work.
#led.cuda()
led = led.to("cuda")
led_enhancement = led('./docs/example.jpeg')[0]
imageio.imwrite('enhanced.jpeg', led_enhancement)
