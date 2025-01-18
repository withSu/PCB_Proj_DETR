import os
import json
import random
from shutil import copy2

def split_dataset(image_dir, annotations_file, train_ratio, output_dir):
    # 경로 설정
    train_dir = os.path.join(output_dir, "train_images")
    val_dir = os.path.join(output_dir, "val_images")
    annotations_dir = os.path.join(output_dir, "annotations")  # 어노테이션 저장 디렉토리
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)  # annotations 디렉토리 생성

    # 전체 이미지 리스트
    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # 학습/검증 데이터 분리
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # 이미지 분리 후 파일 이동
    for img in train_images:
        copy2(os.path.join(image_dir, img), train_dir)
    for img in val_images:
        copy2(os.path.join(image_dir, img), val_dir)

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")

    # 기존 COCO JSON 로드
    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    # COCO JSON 생성
    def create_coco_json(image_list, image_dir, output_json):
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": annotations["categories"]
        }
        image_id_map = {}  # 이미지 ID 매핑

        # 이미지 정보와 어노테이션 추가
        for new_id, file_name in enumerate(image_list, start=1):
            # 이미지 정보
            for img in annotations["images"]:
                if img["file_name"] == file_name:
                    new_image = img.copy()
                    new_image["id"] = new_id
                    coco_format["images"].append(new_image)
                    image_id_map[img["id"]] = new_id

            # 어노테이션 정보
            for ann in annotations["annotations"]:
                if ann["image_id"] in image_id_map:
                    if annotations["images"][ann["image_id"] - 1]["file_name"] == file_name:
                        new_ann = ann.copy()
                        new_ann["image_id"] = image_id_map[ann["image_id"]]
                        coco_format["annotations"].append(new_ann)

        # COCO JSON 저장
        with open(output_json, "w") as f:
            json.dump(coco_format, f, indent=4)

    # 학습/검증 JSON 생성
    create_coco_json(train_images, train_dir, os.path.join(annotations_dir, "train.json"))
    create_coco_json(val_images, val_dir, os.path.join(annotations_dir, "val.json"))

# 실행
split_dataset(
    image_dir="/home/a/A_2024_selfcode/PCB_DETR/raw_datasets/1_images",  # 전체 이미지 경로
    annotations_file="/home/a/A_2024_selfcode/PCB_DETR/datasets/output_coco_format.json",  # 전체 COCO JSON 파일 경로
    train_ratio=0.7,  # 학습 데이터 비율 (70%)
    output_dir="/home/a/A_2024_selfcode/PCB_DETR/datasets/"  # 결과 저장 경로
)
