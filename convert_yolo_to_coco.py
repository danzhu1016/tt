import os
import json
from PIL import Image

# 配置路径
images_dir = r'/Dataset/images/val'  # 这是图片文件夹路径
labels_dir = r'/Dataset/labels/val'  # 这是数据集标注文件文件夹路径
output_json_path = 'yolov5/annotations1.json'  # 这是转换后保存json文件路径已经名字


categories = [
  {"id": 1, "name": "Plastic Bottle"},
  {"id": 2, "name": "Face Mask"},
  {"id": 3, "name": "Paper Bag"},
  {"id": 4, "name": "Plastic Cup"},
  {"id": 5, "name": "Paper Cup"},
  {"id": 6, "name": "Cardboard"},
  {"id": 7, "name": "Peel"},
  {"id": 8, "name": "Cans"},
  {"id": 9, "name": "Plastic Wrapper"},
  {"id": 10, "name": "Paperboard"},
  {"id": 11, "name": "Styrofoam"},
  {"id": 12, "name": "Tetra Pack"},
  {"id": 13, "name": "Colored Glass Bottles"},
  {"id": 14, "name": "Plastic Bag"},
  {"id": 15, "name": "Rags"},
  {"id": 16, "name": "Pile of Leaves"},
  {"id": 17, "name": "Glass Bottle"}
]


coco_data = {
    "images": [],
    "annotations": [],
    "categories": categories
}


def yolo_to_coco(images_dir, labels_dir, output_path):
    anno_id = 1 

    for label_file in os.listdir(labels_dir):
    
        image_id, _ = os.path.splitext(label_file)

     
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        with Image.open(image_path) as img:
            width, height = img.size

       
        coco_data["images"].append({
            "file_name": f"{image_id}.jpg",
            "height": height,
            "width": width,
            "id": str(image_id)
        })

       
        with open(os.path.join(labels_dir, label_file), 'r') as file:
            for line in file:
                category_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

               
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                width_bbox = bbox_width * width
                height_bbox = bbox_height * height

               
                coco_data["annotations"].append({
                    "id": anno_id,
                    "image_id": str(image_id),
                    "category_id": int(category_id),  # 假设YOLO类别索引从0开始
                    "bbox": [x_min, y_min, width_bbox, height_bbox],
                    "area": width_bbox * height_bbox,
                    "iscrowd": 0,
                    "segmentation": []
                })
                anno_id += 1

    
    with open(output_path, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)


yolo_to_coco(images_dir, labels_dir, output_json_path)
