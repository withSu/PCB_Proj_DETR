import torch
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from models import build_model
import util.misc as utils
import importlib
importlib.invalidate_caches()
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Updated sys.path:", sys.path)  # 디버깅용 출력

def load_model(checkpoint_path, args, device='cuda'):
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    return model, criterion, postprocessors

def main_eval(checkpoint_path, args):
    device = torch.device(args.device)
    model, criterion, postprocessors = load_model(checkpoint_path, args, device)
    
    # 데이터셋 빌드
    dataset_val = build_dataset(image_set='val', args=args)
    
    # 데이터 로더 생성
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn
    )
    
    # COCO API 생성
    base_ds = get_coco_api_from_dataset(dataset_val)

    # 모델 평가
    test_stats, coco_evaluator = evaluate(
        model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
    )
    print("COCO bbox mAP:", coco_evaluator.coco_eval['bbox'].stats.tolist())

# 실행
if __name__ == '__main__':
    # 학습 때 사용한 인자 설정
    class Args:
        device = 'cuda'
        dataset_file = 'coco'
        coco_path = '../datasets'  # COCO 데이터셋 경로
        batch_size = 2
        num_workers = 2
        output_dir = ''
        want_class = 1
        hidden_dim = 256  # 숨겨진 차원
        position_embedding = 'sine'  # 위치 임베딩 방식
        lr_backbone = 2e-5  # 백본 학습률
        masks = False  # 마스크 사용 여부
        lr = 1e-4  # 전체 학습률
        weight_decay = 1e-4  # 가중치 감쇠
        epochs = 100  # 학습 에포크 수
        backbone = 'resnet50'  
        dilation = False  

    checkpoint_path = '/home/a/A_2024_selfcode/PCB_proj_DETR/0_output_Backup/checkpoint_099.pth'
    main_eval(checkpoint_path, Args())