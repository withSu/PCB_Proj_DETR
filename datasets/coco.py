# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

# 수정: 필요한 것들을 각각 임포트
from datasets.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomSelect,
    RandomResize,
    RandomSizeCrop,
    GaussianBlur,
    ToTensor,
    Normalize,
    ColorJitter,
    # RandomRotation 등등
)



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_transforms(image_set):
    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(p=0.5),
            # 색상 변화
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            # Resize
            RandomSelect(
                RandomResize([400, 500, 600, 700, 800, 900, 1000], max_size=1333),
                Compose([
                    RandomResize([400, 500, 600]),
                    RandomSizeCrop(300, 600),
                    RandomResize([400, 500, 600, 700, 800, 900, 1000], max_size=1333),
                ])
            ),
            GaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0), p=0.5),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif image_set == 'val':
        return Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"unknown image_set {image_set}")



def build(image_set, args):
    root = Path(args.coco_path)  # ./datasets 디렉토리
    assert root.exists(), f'provided COCO path {root} does not exist'
    
    mode = 'instances'
    PATHS = {
        "train": (root / "train_images", root / "annotations" / "train.json"),
        "val": (root / "val_images", root / "annotations" / "val.json"),
        "test": (root / "test_images", root / "annotations" / "test.json"),
    }

    # image_set이 train, val, test 중 하나인지 확인
    assert image_set in PATHS, f"unknown image_set {image_set}. Valid options are 'train', 'val', 'test'."

    # 이미지와 어노테이션 경로 설정
    img_folder, ann_file = PATHS[image_set]
    print(f"Image folder: {img_folder}")
    print(f"Annotation file: {ann_file}")
    
    # 경로가 존재하는지 확인
    assert img_folder.exists(), f'Image folder {img_folder} does not exist.'
    assert ann_file.exists(), f'Annotation file {ann_file} does not exist.'

    # 데이터셋 로드
    dataset = CocoDetection(
        img_folder, 
        ann_file, 
        transforms=make_coco_transforms(image_set), 
        return_masks=args.masks
    )
    return dataset