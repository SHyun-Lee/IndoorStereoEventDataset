import os
import torch

from pathlib import Path
import numpy as np
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import builtin_meta
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
import matplotlib.pyplot as plt

def process_images_in_directory(input_directory, output_directory):
    # Check if the output directory exists, if not, create it
    os.makedirs(output_directory, exist_ok=True)

    # List all PNG files in the input directory
    png_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    # Set up detectron2 configuration
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("/home/coraldl/meta/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.WEIGHTS = "/home/coraldl/meta/Mask2Former/fine_10000.pth/model_final.pth"
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)
    metadata = builtin_meta._get_builtin_metadata("coco_panoptic_standard")
    coco_metadata = MetadataCatalog.get("coco_panoptic_seg_train").set(**metadata)
    #coco_metadata = MetadataCatalog.get("coco_2017_train_panoptic")
   # MetadataCatalog.get("coco_panoptic_seg_train").thing_classes = ["person", "tv", "dining table", "chair", "bench", "table-merged", "couch" ]
#MetadataCatalog.get("coco_panoptic_seg_train").stuff_classes = ["wall-other-merged", "window-other", "ceiling-merged", "floor-other-merged", "door-stuff", "banner", "light", "counter", "stairs", "cardboard", "cabinet-merged" ]
    for png_file in png_files:
        # Build the full path for each PNG file
        image_path = os.path.join(input_directory, png_file)

        # Process and visualize the image
        im = cv2.imread(str(image_path))
        predictor.device = torch.device("cuda")
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
                                              outputs["panoptic_seg"][1]).get_image()
        v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()

        # Create subdirectories for each result type
        panoptic_output_directory = os.path.join(output_directory, "panoptic")
        instance_output_directory = os.path.join(output_directory, "instance")
        semantic_output_directory = os.path.join(output_directory, "semantic")
        os.makedirs(panoptic_output_directory, exist_ok=True)
        os.makedirs(instance_output_directory, exist_ok=True)
        os.makedirs(semantic_output_directory, exist_ok=True)

        # Save the result images to their respective subdirectories
        panoptic_result_filename = f"{png_file}"
        instance_result_filename = f"{png_file}"
        semantic_result_filename = f"{png_file}"

        panoptic_result_image_path = os.path.join(panoptic_output_directory, panoptic_result_filename)
        instance_result_image_path = os.path.join(instance_output_directory, instance_result_filename)
        semantic_result_image_path = os.path.join(semantic_output_directory, semantic_result_filename)

        cv2.imwrite(panoptic_result_image_path, panoptic_result)
        cv2.imwrite(instance_result_image_path, instance_result)
        cv2.imwrite(semantic_result_image_path, semantic_result)
        
# Example usage
input_directory = "/home/coraldl/Downloads/2024-1-5_14-54-47/webcam/images"
output_directory = "/home/coraldl/Pictures/new_fine_10000"
process_images_in_directory(input_directory, output_directory)

