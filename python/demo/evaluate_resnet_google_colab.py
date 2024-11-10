import os
os.chdir('/home/jknize/main/repo/CSC578/detectron2')
print(os.getcwd())

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_val_dataset", {}, "../datasets/coco/annotations/filtered_instances_val2017.json", "../datasets/coco/val2017")

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2
import matplotlib.pyplot as plt

my_dataset_metadata = MetadataCatalog.get("my_val_dataset")
dataset_dicts = DatasetCatalog.get("my_val_dataset")

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# load simple ResNet-50 model without FPN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("configs/COCOInstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("configs/COCOInstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_val_dataset", cfg, False, output_dir="./knize/output")
val_loader = build_detection_test_loader(cfg, "my_val_dataset")

metrics = inference_on_dataset(DefaultTrainer.build_model(cfg), val_loader, evaluator)
print("Evaluation Results:", metrics)