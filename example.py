import cv2
from led.pipelines.led_pipeline import LEDPipeline
led = LEDPipeline.from_pretrained('pretrained_weights')
led.to('cuda')
led_enhancement = led('./docs/example.jpeg')[0]
cv2.imwrite('enhanced_example.jpeg', led_enhancement[:, :, ::-1])
