import re
import matplotlib.pyplot as plt

def parse_epoch_log(file_path):
    epochs = []
    train_losses = []
    test_losses = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Train Loss 추출
        if "Train Stats" in line:
            epoch = int(re.search(r"Epoch (\d+)", line).group(1))
            train_loss = float(re.search(r"'loss': ([\d\.]+)", line).group(1))
            epochs.append(epoch)
            train_losses.append(train_loss)

        # Test Loss 추출
        elif "Test  Stats" in line:
            test_loss = float(re.search(r"'loss': ([\d\.]+)", line).group(1))
            test_losses.append(test_loss)

    return epochs, train_losses, test_losses


def plot_losses(epochs, train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", linestyle="-")
    plt.plot(epochs, test_losses, label="Validation Loss", marker="o", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# 로그 파일 경로 설정
log_file = "../output_augmented_eos0.2/epoch_log.txt"

# 로그 파일 파싱
epochs, train_losses, test_losses = parse_epoch_log(log_file)

# 손실 그래프 출력
plot_losses(epochs, train_losses, test_losses)
