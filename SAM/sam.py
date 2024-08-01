# mkdir SAM
# cd SAM
# mkdir weights
# cd weights
# wget -q 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

import cv2
import numpy as np
from segment_anything import SamPredictor
from segment_anything import sam_model_registry

class SAM():
    def __init__(self):
        self.sam = sam_model_registry["vit_h"](checkpoint="SAM/weights/sam_vit_h_4b8939.pth")

    def get_mask(self, image_rgb, box):
        mask_predictor = SamPredictor(self.sam)
        mask_predictor.set_image(image_rgb)
        masks, _, _ = mask_predictor.predict(
            box=box,
            multimask_output=False
        )
        return masks[0]
