import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from util.misc import resize_boxes_to_original


# 1. COCO 어노테이션 및 평가 도구 초기화
# 임의의 어노테이션 데이터 생성 (in-memory)
annotations = {
    "images": [
        {
            "id": 264,
            "file_name": "dummy_image.jpg",
            "width": 3904,
            "height": 3904,
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 264,
            "category_id": 1,
            "bbox": [1351, 1151, 1829, 1823],  # [x_min, y_min, width, height] #GT에서 설정된 bbox: [1351, 1151, 1829, 1823]
            "area": 3334883, #면적(area): 1829 * 1823 = 3334883
            "iscrowd": 0,
        }
        
    ],
    "categories": [{"id": 1, "name": "object"}],
}
coco_gt = COCO()
coco_gt.dataset = annotations
coco_gt.createIndex()

# 2. 예측 데이터 생성
predictions = {
    264: {
        "boxes": torch.tensor([[277, 236, 656, 653]]),  # 리사이즈된 bbox 좌표
        "scores": torch.tensor([0.9]),  # 신뢰도
        "labels": torch.tensor([1]),  # 클래스 라벨
        "size": (800, 800),  # 리사이즈된 이미지 크기
        "orig_size": (3904, 3904),  # 원본 이미지 크기
    }
}

#현재 GT의 area는 3334883으로 Large 크기에 해당하며, COCO 평가에서도 Large 영역에 포함된 것으로 보인다. 이는 원본 크기 기준으로 면적이 계산되었음을 의미한다.



# 3. BBox를 원본 크기로 복원
for image_id, pred in predictions.items():
    pred["boxes"] = resize_boxes_to_original(
        pred["boxes"].clone(), pred["size"], pred["orig_size"]
    )

# 4. COCO 평가를 위한 결과 변환
coco_results = []
for image_id, pred in predictions.items():
    boxes = pred["boxes"]
    scores = pred["scores"].tolist()
    labels = pred["labels"].tolist()
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box.tolist()
        width = x_max - x_min
        height = y_max - y_min
        coco_results.append(
            {
                "image_id": image_id,
                "category_id": labels[i],
                "bbox": [x_min, y_min, width, height],
                "score": scores[i],
            }
        )

# 5. COCO API를 사용하여 평가
coco_dt = coco_gt.loadRes(coco_results)
evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
