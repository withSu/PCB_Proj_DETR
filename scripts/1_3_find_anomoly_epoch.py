import re

# 로그 파일 경로
log_file_path = "/home/user/Desktop/Vision-team/KBS/PCB_proj_DETR/output_augmented_eos0.2/epoch_log.txt"

def parse_epoch_log(file_path):
    epoch_loss_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"Epoch (\d+) - Train Stats: .*'loss': ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epoch_loss_dict[epoch] = loss  # 같은 에포크가 여러 번 나오면 최신 값으로 덮어씀

    # 에포크 번호 기준으로 정렬
    sorted_epochs = sorted(epoch_loss_dict.keys())
    sorted_losses = [epoch_loss_dict[epoch] for epoch in sorted_epochs]

    return sorted_epochs, sorted_losses

def find_spike_epochs(epochs, losses, threshold=1.5):
    spike_epochs = []
    for i in range(1, len(losses)):
        loss_change = losses[i] - losses[i - 1]  # 다음 에포크로 넘어갈 때 loss 증가량
        if loss_change > threshold:
            spike_epochs.append((epochs[i - 1], losses[i - 1], epochs[i], losses[i], loss_change))

    return spike_epochs

# 로그 파일에서 에포크 및 손실 값 추출
epochs, losses = parse_epoch_log(log_file_path)

# 손실이 급격히 증가한 에포크 찾기 (기본 임계값: 1.5)
spike_epochs = find_spike_epochs(epochs, losses)

# 결과 출력
for prev_epoch, prev_loss, epoch, loss, change in spike_epochs:
    print(f"Epoch {prev_epoch} -> Epoch {epoch}: Loss {prev_loss} -> {loss} (Increase: {change})")
