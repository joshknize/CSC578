from pycocotools.coco import COCO
import requests
import os
import shutil

coco = COCO('repo/CSC578/datasets/coco/annotations/filtered_instances_val2017_2.json')

catIds = coco.getCatIds(catNms=['dog'])

imgIds = []
for cat in catIds:
    imgIds.extend(coco.getImgIds(catIds=cat))
imgIds = list(set(imgIds))
images = coco.loadImgs(imgIds)

for im in images:
    img_file = im['file_name']
    source_path = 'repo/CSC578/datasets/coco/val2017_subset/' + img_file
    dest_path = 'repo/CSC578/datasets/coco/val2017_dog/' + img_file
    shutil.copy(source_path, dest_path)