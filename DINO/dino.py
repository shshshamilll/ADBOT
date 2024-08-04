import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class DINO():
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

    def get_box(self, image_for_dino, object):
        inputs = self.processor(images=image_for_dino, text=object, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image_for_dino.size[::-1]]
        )
        return result[0]["boxes"].numpy()
