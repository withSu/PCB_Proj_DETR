import torch
import torchvision.transforms as T
import cv2
import numpy as np

# DETR 모델 정의부 (간단히 예시로 작성)
# 실제론 DETR 공식 레포나, 직접 작성한 모델 파이썬 코드에서 가져올 수도 있다.
# 여기서는 시연을 위해 매우 단순화된 형태로 작성한다.
class DetrModel(torch.nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        # 학습된 백본과 트랜스포머 등을 포함하고 있다고 가정한다
        # 실제 코드에서는 DETR 전체 구조가 구현되어 있어야 한다.

    def forward(self, images):
        # 예시로만 간단히 랜덤 값 반환
        # 실제론 TransformerEncoder, Decoder 등을 거쳐
        # 바운딩박스(좌표)와 클래스 확률을 예측한다.
        batch_size = images.shape[0]
        # [batch, num_boxes, 4] 형태로 좌표 예측 값이 나온다고 가정
        # [batch, num_boxes, num_classes] 형태로 클래스 logits가 나온다고 가정
        pred_boxes = torch.rand(batch_size, 10, 4)  # 예시: 10개의 박스
        pred_logits = torch.rand(batch_size, 10, 91)  # 예시: COCO 클래스 91개
        return pred_logits, pred_boxes

# 후처리 함수 (NMS 없이 간단히 예시)
# 실제론 DETR에서 제공하는 post_process 함수를 사용하거나
# 점수(신뢰도) threshold, NMS 등을 적용해 박스 최종 선별을 해야 한다.
def post_process(outputs, score_thresh=0.5):
    pred_logits, pred_boxes = outputs
    # 시연을 위해 단순히 첫 번째 이미지만 사용
    logits = pred_logits[0]
    boxes = pred_boxes[0]

    # softmax로 클래스 확률 구한다
    probs = torch.softmax(logits, dim=-1)  
    # 맨 마지막 클래스(Background 등)를 제외한 최대값과 인덱스 구한다
    scores, labels = probs[..., :-1].max(dim=-1)  

    # 점수 threshold 이상인 인덱스만 추출
    keep = scores > score_thresh
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    # x, y, w, h 형태를 x1, y1, x2, y2 형태로 변환
    # DETR은 기본적으로 [center_x, center_y, width, height] 형태를 사용하는 예시가 많다.
    # 여기서는 [x, y, w, h]라고 가정한다.
    # 실제 DETR 구현에서는 center_x, center_y 등 변환에 주의해야 한다.
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.0  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.0  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.0  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.0  # y2

    return boxes_xyxy, labels, scores

def visualize(image_path, model, device='cpu'):
    model.eval()
    model.to(device)

    # 이미지 읽기
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        print("이미지를 불러오지 못했다.")
        return
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # DETR에 맞는 전처리
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((800, 800)),  # 예시로 크기 고정
        T.ToTensor()
    ])
    # 이미지 numpy -> tensor 변환
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 모델 추론
    outputs = model(img_tensor)

    # 후처리
    boxes_xyxy, labels, scores = post_process(outputs, score_thresh=0.5)

    # 바운딩박스 시각화
    # cv2는 좌표가 정수여야 하므로 int 변환
    for box, label, score in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
        class_id = label.item()
        conf = score.item()

        # 박스 그리기
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Class: {class_id}, Conf: {conf:.2f}"
        cv2.putText(orig_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # 시각화 결과 보여주기
    cv2.imshow("DETR Visualization", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 학습된 모델 checkpoint 로드
    checkpoint_path = "checkpoint_099.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 예시용 간단 모델 정의 (실제론 DETR 구조가 구현된 클래스여야 한다)
    model = DetrModel(num_classes=91)

    # 저장된 파이썬 모델을 불러온 뒤 state_dict만 로드 (예시)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 시각화할 이미지 경로 (원하는 이미지로 변경)
    test_image_path = "test_image.jpg"

    # 시각화 함수 호출
    visualize(test_image_path, model, device=device)

if __name__ == "__main__":
    main()
