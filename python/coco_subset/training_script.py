import os
os.chdir('/home/jknize/main/repo/CSC578/detectron2')
print(os.getcwd())

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import torch

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

register_coco_instances("coco_train_subset", {}, "../datasets/coco/annotations/filtered_instances_train2017_2.json", "../datasets/coco/train2017_subset")
register_coco_instances("coco_val_subset", {}, "../datasets/coco/annotations/filtered_instances_val2017_2.json", "../datasets/coco/val2017_subset")

my_dataset_metadata = MetadataCatalog.get("coco_train_subset")
my_dataset_metadata.thing_classes = ["person", "dog", "bottle", "chair", "book"]
dataset_dicts = DatasetCatalog.get("coco_train_subset")

# load simple ResNet-50 model without FPN
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") #ImageNet pre-trained
cfg.OUTPUT_DIR = "knize/output/subset"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

cfg.DATASETS.TRAIN = ("coco_train_subset",)
cfg.DATASETS.TEST = ("coco_val_subset",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

# solver config (guesses)
cfg.SOLVER.IMS_PER_BATCH = 32
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.SOLVER.STEPS = [] # disable learning decay

# run on GPU
cfg.MODEL.DEVICE = 'cuda'

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.model.to(cfg.MODEL.DEVICE)

trainer.train()

# evaluate performance of trained subset model
cfg.MODEL.WEIGHTS = "knize/output/subset/model_final.pth"

trainer.model.eval()

evaluator = COCOEvaluator("coco_val_subset", ("bbox",), False, output_dir="./knize/output/subset/trained")
val_loader = build_detection_test_loader(cfg, "coco_val_subset")
print(inference_on_dataset(trainer.model, val_loader, evaluator))