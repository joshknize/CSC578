import json
import os

os.chdir('/home/jknize/main/repo/CSC578/detectron2')

chosen_class_ids = [1, 18, 44, 62, 84]  # (person, dog, bottle, chair, book)

annotations_path = '../datasets/coco/annotations/instances_train2017.json'

# Load the annotations
with open(annotations_path, 'r') as f:
    coco_annotations = json.load(f)

# Filter annotations for the chosen classes
filtered_annotations = []
image_ids_for_filtered = set()  # To track which images have annotations for the chosen classes
for annotation in coco_annotations['annotations']:
    if annotation['category_id'] in chosen_class_ids:
        filtered_annotations.append(annotation)
        image_ids_for_filtered.add(annotation['image_id'])

# Filter the categories as well
filtered_categories = [category for category in coco_annotations['categories'] if category['id'] in chosen_class_ids]

# Filter the images to include only those that have annotations for the chosen classes
filtered_images = [image for image in coco_annotations['images'] if image['id'] in image_ids_for_filtered]

# Update the original coco json with the filtered annotations
coco_annotations['annotations'] = filtered_annotations
coco_annotations['categories'] = filtered_categories
coco_annotations['images'] = filtered_images

# Save the filtered annotations to a new file
filtered_annotations_path = '../datasets/coco/annotations/filtered_instances_train2017_2.json'
with open(filtered_annotations_path, 'w') as f:
    json.dump(coco_annotations, f)


