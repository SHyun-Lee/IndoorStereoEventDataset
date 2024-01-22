from detectron2.data.datasets import register_coco_panoptic_separated, load_sem_seg, register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import matplotlib.pyplot as plt
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import builtin_meta
import sys
import os
import json



dataset_name = "custom_panoptic_dataset"

image_root = "/home/coraldl/meta/Mask2Former/datasets/coco/train_RGB/"
panoptic_root = "/home/coraldl/meta/Mask2Former/datasets/coco/train_mask"
panoptic_json = "/home/coraldl/meta/Mask2Former/datasets/coco/annotations/d/train.json"
instances_json = "/home/coraldl/meta/Mask2Former/datasets/coco/annotations/instance_train.json"
sem_seg_root = "/home/coraldl/meta/Mask2Former/datasets/coco/panoptic_semseg_train"
# 데이터셋 등록
sepa = "custom_panoptic_dataset_separated"
register_coco_instances(
     name = "coco_instance_train",
     metadata = {},
     json_file ="/home/coraldl/meta/Mask2Former/datasets/coco_instance/annotations/instance_train.json",
     image_root ="/home/coraldl/meta/Mask2Former/datasets/coco_instance/train_RGB")
     
register_coco_instances(
     name = "coco_instance_val",
     metadata = {},
     json_file ="/home/coraldl/meta/Mask2Former/datasets/coco_instance/annotations/instance_val.json",
     image_root ="/home/coraldl/meta/Mask2Former/datasets/coco_instance/val_RGB")
my_dataset_train_metadata = MetadataCatalog.get("coco_instance_train")
dataset_dicts = DatasetCatalog.get("coco_instance_train")
# 데이터셋이 비어있지 않은지 확인
if len(dataset_dicts) > 0:
    # 데이터셋 시각화
    for d in random.sample(dataset_dicts, 3):    
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img_rgb = vis.get_image()[:, :, ::-1]
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(d["file_name"])
        plt.show()
else:
    print("데이터셋이 비어있습니다.")

