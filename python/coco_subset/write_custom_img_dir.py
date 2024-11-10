from pycocotools.coco import COCO
import requests
import os
import shutil

coco = COCO('repo/CSC578/datasets/coco/annotations/instances_train2017.json')

catIds = coco.getCatIds(catNms=['person', 'dog', 'chair', 'bottle', 'book'])

imgIds = []
for cat in catIds:
    imgIds.extend(coco.getImgIds(catIds=cat))
imgIds = list(set(imgIds))
images = coco.loadImgs(imgIds)

for im in images:
    img_file = im['file_name']
    source_path = 'repo/CSC578/datasets/coco/train2017/' + img_file
    dest_path = 'repo/CSC578/datasets/coco/train2017_subset/' + img_file
    shutil.copy(source_path, dest_path)