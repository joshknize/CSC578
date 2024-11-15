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

register_coco_instances("coco_train_dog", {}, "../datasets/coco/annotations/dog_instances_train2017.json", "../datasets/coco/train2017_dog")
register_coco_instances("coco_val_dog", {}, "../datasets/coco/annotations/dog_instances_val2017.json", "../datasets/coco/val2017_dog")

my_dataset_metadata = MetadataCatalog.get("coco_train_dog")
my_dataset_metadata.thing_classes = ["dog"]
dataset_dicts = DatasetCatalog.get("coco_train_dog")

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml") #ImageNet pre-trained
cfg.OUTPUT_DIR = "knize/output/dog20241115"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

cfg.DATASETS.TRAIN = ("coco_train_dog",)
cfg.DATASETS.TEST = ("coco_val_dog",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 50000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 1000

# run on GPU
cfg.MODEL.DEVICE = 'cuda'

trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
trainer.model.to(cfg.MODEL.DEVICE)

trainer.train()

# evaluate performance of trained subset model
cfg.MODEL.WEIGHTS = "knize/output/dog_resnet_test/model_final.pth"

trainer.model.eval()

evaluator = COCOEvaluator("coco_val_subset", ("bbox",), False, output_dir="./knize/output/dog/resnet_test")
val_loader = build_detection_test_loader(cfg, "coco_val_dog")
print(inference_on_dataset(trainer.model, val_loader, evaluator))