import os
import torch
import cv2

def inference_and_visualize(model, postprocessors, data_loader, device, output_dir):
    """
    (1) data_loader 순회하며 모델 추론
    (2) 결과를 이미지에 시각화
    (3) output_dir/vis 에 저장
    """
    model.eval()
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)  # 디렉토리 생성 확인
    print(f"[INFO] Visualization directory: {vis_dir}")

    print("Starting final inference & visualization on the entire dataset...")

    for samples, targets in data_loader:
        samples = samples.to(device)
        with torch.no_grad():
            outputs = model(samples)

        # postprocess
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        for i, result in enumerate(results):
            image_id = targets[i]["image_id"].item()

            # COCO 정보에서 파일명 얻기 (dataset.coco.loadImgs)
            img_info = data_loader.dataset.coco.loadImgs([image_id])[0]
            file_name = img_info["file_name"]  # 예: "image1.jpg"
            image_path = os.path.join("./datasets/val_images", file_name)

            print(f"[INFO] Processing image: {image_path}")

            visualize_single_image(
                image_path,
                result,
                vis_dir,
                threshold=0.5
            )


def visualize_single_image(image_path, result, vis_dir, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to open image: {image_path}")
        return

    boxes  = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    labels = result["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label}:{score:.2f}"
        cv2.putText(img, text, (x1, max(y1-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    base_name = os.path.basename(image_path)  # ex) "image1.jpg"
    out_name = base_name.replace(".jpg", "_res.jpg")
    save_path = os.path.join(vis_dir, out_name)

    os.makedirs(vis_dir, exist_ok=True)
    print(f"[INFO] Saving visualization to: {save_path}")

    success = cv2.imwrite(save_path, img)
    if not success:
        print(f"[ERROR] Failed to save: {save_path}")
