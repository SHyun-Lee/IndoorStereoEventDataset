import os
import torch

from pathlib import Path
import numpy as np
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
import matplotlib.pyplot as plt

register_coco_instances(
    name="my_d_v",
    metadata={},
    json_file="/home/coraldl/meta/Mask2Former/datasets/mydata/annotations/val.json",
    image_root="/home/coraldl/meta/Mask2Former/datasets/mydata/val"
)

register_coco_instances(
    name="my_d_t",
    metadata={},
    json_file="/home/coraldl/meta/Mask2Former/datasets/mydata/annotations/train.json",
    image_root="/home/coraldl/meta/Mask2Former/datasets/mydata/val"
)
def process_images_in_directory(input_directory, output_directory):
    # Check if the output directory exists, if not, create it
    os.makedirs(output_directory, exist_ok=True)

    # List all PNG files in the input directory
    png_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    # Set up detectron2 configuration
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.WEIGHTS = "~/weights1.pth/model_final.pth"
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)
    coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

    for png_file in png_files:
        # Build the full path for each PNG file
        image_path = os.path.join(input_directory, png_file)

        # Process and visualize the image
        im = cv2.imread(str(image_path))
        predictor.device = torch.device("cuda")
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        result = semantic_result[:, :, ::-1]

        # Save the result image to the output directory
        result_filename = f"result_{png_file}"
        result_image_path = os.path.join(output_directory, result_filename)
        cv2.imwrite(result_image_path, result)

        # Visualize using Matplotlib
        print(f"Processed and saved: {image_path} -> {result_image_path}")

# Example usage
input_directory = "/home/coral/Downloads/prime1/right/RGB"
output_directory = "/home/coral/Downloads/outprime/right/RGB"
process_images_in_directory(input_directory, output_directory)

