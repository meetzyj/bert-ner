import re
import matplotlib.pyplot as plt

# 读取日志文件
log_file = 'training2.log'
with open(log_file, 'r') as f:
    log_lines = f.readlines()

# 提取 epoch 和学习率信息
epochs = []
learning_rates = []
for line in log_lines:
    match = re.search(r'Epoch: (\d+), .* LR: ([\d\.]+)', line)
    if match:
        epoch = int(match.group(1))
        lr = float(match.group(2))
        epochs.append(epoch)
        learning_rates.append(lr)

import pdb; pdb.set_trace()
# 绘制 epoch-LR 变化曲线
plt.plot(epochs, learning_rates, 'b', label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Epoch vs Learning Rate')
plt.legend()
plt.grid(True)
plt.savefig('epoch_lr_curve.png')  # 保存图片