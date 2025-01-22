import os
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from models.detr import build  # build_detr 대신 build 함수를 가져옴
import argparse  # argparse 모듈 추가

# main.py에서 가져온 get_args_parser 함수
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--want_class', type=int, help='number of class which want to finetuning')  # 추가

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_cost_class', default=1, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")

    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

# 학습된 모델 로드 함수
def load_model(weights_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # args 객체 생성
    parser = get_args_parser()
    args = parser.parse_args([])  # 빈 리스트를 전달하여 기본값으로 args 객체 생성
    args.num_classes = num_classes  # 클래스 수 설정
    args.want_class = num_classes  # want_class도 설정 (추가된 부분)
    args.device = device  # 디바이스 설정

    # DETR 모델 빌드
    model, _, _ = build(args)  # build 함수 사용
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)  # 가중치 로드
    model.to(device)
    model.eval()  # 평가 모드로 전환
    return model

def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    x_min = x_c - 0.5 * w
    y_min = y_c - 0.5 * h
    x_max = x_c + 0.5 * w
    y_max = y_c + 0.5 * h
    return torch.stack((x_min, y_min, x_max, y_max), dim=-1)

def rescale_bboxes(boxes, size):
    img_w, img_h = size
    boxes = boxes.clone()
    boxes[:, 0::2] *= img_w
    boxes[:, 1::2] *= img_h
    return boxes

def visualize(image_path, model, threshold=0.7, save_dir="/home/a/A_2024_selfcode/PCB_proj_DETR/0_output_Backup/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 원본 이미지 열기
    image = Image.open(image_path).convert("RGB")
    img_size = image.size

    with torch.inference_mode():
        # 텐서 변환 및 모델 예측
        transform = F.to_tensor(image).unsqueeze(0).to(device)
        outputs = model(transform)
        logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]

        # 바운딩 박스 변환 및 필터링
        prob = logits.softmax(-1)[:, :-1]
        scores, labels = prob.max(-1)
        keep = scores > threshold
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
        pred_boxes = rescale_bboxes(pred_boxes, img_size)

    # 필터링된 결과
    pred_boxes = pred_boxes[keep]
    scores = scores[keep].tolist()

    # 바운딩 박스 그리기
    draw = ImageDraw.Draw(image)
    for box, score in zip(pred_boxes.tolist(), scores):
        x_min, y_min, x_max, y_max = box
        x_min, x_max = sorted([max(0, x_min), min(x_max, img_size[0])])
        y_min, y_max = sorted([max(0, y_min), min(y_max, img_size[1])])
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min), f"{score:.2f}", fill="red")

    # 결과 저장
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, os.path.basename(image_path))
    image.save(output_path)
    print(f"Saved visualization to {output_path}")

    # 캐시 정리
    del transform, outputs, logits, pred_boxes, prob, keep
    torch.cuda.empty_cache()


# 폴더 내 모든 이미지 순회
def process_image_folder(folder_path, model, threshold=0.7, save_dir=None):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        visualize(image_path, model, threshold, save_dir)

# 실행
if __name__ == "__main__":
    weights_path = "/home/a/A_2024_selfcode/PCB_proj_DETR/0_output_Backup/checkpoint_099.pth"  # 학습된 가중치 경로
    num_classes = 1  # 클래스 수 (no-object 포함)
    test_images_folder = "../datasets/test_images"  # 테스트 이미지 폴더 경로
    save_visualizations_dir = "../1_output_visualizations"  # 결과 저장 폴더 (None으로 설정 시 화면 표시)

    # 모델 로드
    model = load_model(weights_path, num_classes)

    # 이미지 폴더 처리
    process_image_folder(test_images_folder, model, threshold=0.7, save_dir=save_visualizations_dir)