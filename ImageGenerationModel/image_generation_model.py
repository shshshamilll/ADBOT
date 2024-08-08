from diffusers import ControlNetModel
import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline

class ImageGenerationModel():
    def __init__(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-sdxl-1.0",
            torch_dtype=torch.float16
        )
        self.inpaint_pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            controlnet=controlnet
        )
        self.inpaint_pipeline.enable_model_cpu_offload()
        self.image_to_image_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16
        )
        self.image_to_image_pipeline.enable_model_cpu_offload()

    def generate(self, prompt, image_for_image_generation_model, mask, control, number, strength):
        image_after_inpaint_pipeline = self.inpaint_pipeline(
            prompt=prompt,
            image=image_for_image_generation_model,
            mask_image=mask,
            control_image=control,
            negative_prompt="low quality"
        ).images[0]
        image_after_image_to_image_pipeline = self.image_to_image_pipeline(
            prompt="high quality",
            image=image_after_inpaint_pipeline,
            negative_prompt="low quality",
            strength=strength
        ).images[0]
        image_after_image_to_image_pipeline.save(f"GeneratedImages/generated_image_{number}.png")
