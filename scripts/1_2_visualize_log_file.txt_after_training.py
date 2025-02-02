import re
import matplotlib.pyplot as plt

def parse_epoch_log(file_path):
    epoch_loss_dict = {}  # 최신 Train Loss 저장
    epoch_test_loss_dict = {}  # 최신 Test Loss 저장

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Train Loss 추출
        if "Train Stats" in line:
            match = re.search(r"Epoch (\d+) - Train Stats: .*'loss': ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                epoch_loss_dict[epoch] = train_loss  # 최신 값으로 덮어씀

        # Test Loss 추출
        elif "Test  Stats" in line:
            match = re.search(r"Epoch (\d+) - Test  Stats: .*'loss': ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                test_loss = float(match.group(2))
                epoch_test_loss_dict[epoch] = test_loss  # 최신 값으로 덮어씀

    # 에포크 번호 기준 정렬
    sorted_epochs = sorted(epoch_loss_dict.keys())
    sorted_train_losses = [epoch_loss_dict[epoch] for epoch in sorted_epochs]
    sorted_test_losses = [epoch_test_loss_dict.get(epoch, None) for epoch in sorted_epochs]  # 없는 경우 None

    return sorted_epochs, sorted_train_losses, sorted_test_losses


def plot_losses(epochs, train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", linestyle="-")

    # Test Loss가 없는 경우를 대비하여 필터링
    valid_test_losses = [t for t in test_losses if t is not None]
    valid_test_epochs = [epochs[i] for i, t in enumerate(test_losses) if t is not None]

    plt.plot(valid_test_epochs, valid_test_losses, label="Validation Loss", marker="o", linestyle="--")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# 로그 파일 경로 설정
log_file = "/home/user/Desktop/Vision-team/KBS/PCB_proj_DETR/output_backup/0_output_augmented_eos0.2(basic wight, batch=8, epoch300)/epoch_log.txt"

# 로그 파일 파싱
epochs, train_losses, test_losses = parse_epoch_log(log_file)

# 손실 그래프 출력
plot_losses(epochs, train_losses, test_losses)
