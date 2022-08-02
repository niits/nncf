from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import torch
from PIL.JpegImagePlugin import JpegImageFile
from torch import Tensor


class CollateFn:
    def __init__(self, transform: A.Compose) -> Dict:
        self.transform = transform

    def __call__(self, data: List[Tuple[JpegImageFile, int]]) -> Tuple[Tensor, Tensor]:
        return {
            "image": torch.stack(
                [
                    self.transform(image=np.array(element["image"].convert("RGB")))[
                        "image"
                    ]
                    for element in data
                ]
            ).float(),
            "label": torch.Tensor([element["label"] for element in data]).long(),
        }
