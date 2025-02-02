import os
import cv2
import numpy as np

def compare_images(folder1, folder2, display_width=800):
    images1 = set(os.listdir(folder1))
    images2 = set(os.listdir(folder2))
    
    common_images = images1.intersection(images2)
    
    if not common_images:
        print("공통된 이미지가 없습니다.")
        return
    
    for img_name in sorted(common_images):
        img1_path = os.path.join(folder1, img_name)
        img2_path = os.path.join(folder2, img_name)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"이미지를 불러올 수 없습니다: {img_name}")
            continue
        
        height = max(img1.shape[0], img2.shape[0])
        width1, width2 = img1.shape[1], img2.shape[1]
        
        img1_resized = cv2.resize(img1, (width1, height))
        img2_resized = cv2.resize(img2, (width2, height))
        
        combined = np.hstack((img1_resized, img2_resized))
        
        scale_factor = 2.0 * display_width / combined.shape[1]  # 2배 확대 적용
        new_width = int(combined.shape[1] * scale_factor)
        new_height = int(combined.shape[0] * scale_factor)
        resized_combined = cv2.resize(combined, (new_width, new_height))
        
        # 이미지에 파일 경로 추가 (검은색 바탕 추가)
        padding = 50  # 경로를 넣을 공간 확보
        img_with_text = np.zeros((resized_combined.shape[0] + padding, resized_combined.shape[1], 3), dtype=np.uint8)
        img_with_text[:resized_combined.shape[0], :, :] = resized_combined
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)  # 흰색 글씨
        
        # 경로를 두 줄로 나누어 표시
        cv2.putText(img_with_text, f"Left: {img1_path}", (10, resized_combined.shape[0] + 20), font, font_scale, text_color, font_thickness)
        cv2.putText(img_with_text, f"Right: {img2_path}", (10, resized_combined.shape[0] + 40), font, font_scale, text_color, font_thickness)

        cv2.imshow(f"Comparison: {img_name}", img_with_text)

        while True:
            key = cv2.waitKey(1)  # 사용자의 키 입력 대기
            if key == 27 or cv2.getWindowProperty(f"Comparison: {img_name}", cv2.WND_PROP_VISIBLE) < 1:
                break  # ESC 키를 누르거나 창을 닫으면 다음 이미지로 이동
        
        cv2.destroyAllWindows()  # 현재 창 닫기

    cv2.destroyAllWindows()

# 사용 예시
folder1 = "/home/a/A_2024_selfcode/PCB_proj_DETR/3_output_Backup(300_, batch=2)(eos 0.2)_1.30/vis"
folder2 = "/home/a/A_2024_selfcode/PCB_proj_DETR/4_ouput (300_, batch=2)(eos 0.4)_1.31/vis"
compare_images(folder1, folder2, display_width=800)