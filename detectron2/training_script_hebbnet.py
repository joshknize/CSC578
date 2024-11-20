import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import time

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
# cfg.MODEL.WEIGHTS = "knize/output/dog_ROI_HEAD_tinkering2/model_final.pth"
cfg.merge_from_file("configs/COCO-Detection/hebbnet_backbone.yaml")
cfg.OUTPUT_DIR = "knize/output/dog_ROI_HEAD_tinkering2"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

cfg.DATASETS.TRAIN = ("coco_train_dog",)
cfg.DATASETS.TEST = ("coco_val_dog",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.OUTPUT_LAYER_SIZE = 1
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = .00002
cfg.SOLVER.MAX_ITER = 5500*2 # one epoch = 5500 (with batch size = 1)
# cfg.SOLVER.CHECKPOINT_PERIOD = 1000 # save disk space on server
# cfg.TEST.EVAL_PERIOD = 1000


cfg.INPUT.MIN_SIZE_TRAIN = (128,)
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
cfg.INPUT.MAX_SIZE_TRAIN = 128
cfg.INPUT.MIN_SIZE_TEST = 128
cfg.INPUT.MAX_SIZE_TEST = 128
# cfg.PROPOSAL_GENERATOR: PrecomputedProposals # this may be an option to potentially avoid issues with proposal generation
# TODO: think about image normalization "cfg.PIXEL_MEAN"
cfg.MODEL.ROI_HEADS.IN_FEATURES: ['res4']

# change the ROI HEAD expected channels and conv_dim
    # notes: we default to using Res5ROIHeads. It uses a bottleneck block and a less flexible and seemingly less configurable architecture.
    # "The ROIHeads in a typical "C4" R-CNN model, where the box and mask head share the cropping and the per-region feature computation by a Res5 block." 
    # we should probably opt for a simpler ROI head that is more customizable and doesn't have added complexity due to the mask for image segmentation
cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 8
cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 8
# TODO i'm not sure how to set these two up below (pave)
cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 2
cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
# arbitrarily don't save model checkpoint
cfg.SOLVER.CHECKPOINT_PERIOD = 9999999999999999999999999999999999

# new
cfg.MODEL.NUM_HIDDEN = 8
cfg.MODEL.HEBB_LR = .00000001 # fewest number of zeros "allowed" before gradient explosion

cfg.MODEL.IMG_VIS = False
cfg.MODEL.FEAT_VIS = False
cfg.MODEL.FEAT_VIS_NUM = 0

# run on GPU
cfg.MODEL.DEVICE = 'cuda'

trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
trainer.model.to(cfg.MODEL.DEVICE)
# trainer.model = torch.nn.parallel.DistributedDataParallel(
#     trainer.model,
#     device_ids=[torch.cuda.current_device()],
#     output_device=torch.cuda.current_device()
# )

trainer.train()

trainer.model.eval()

evaluator = COCOEvaluator("coco_val_dog", ("bbox",), False, output_dir="./knize/output/dog_ROI_HEAD_tinkering2")
val_loader = build_detection_test_loader(cfg, "coco_val_dog")
print(inference_on_dataset(trainer.model, val_loader, evaluator))