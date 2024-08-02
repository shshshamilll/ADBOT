import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class DINO():
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

    def get_box(self, image, text):
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        return results[0]["boxes"]
