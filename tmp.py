import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========================
# 日志配置
# ========================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "train.log"),
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ========================
# 数据集定义
# ========================
class RectDataset(Dataset):
    def __init__(self, txt_file, augment=False):
        self.samples = []
        self.augment = augment
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 13:  # 8坐标 + 4mask + id
                    continue
                coords = list(map(float, parts[0:8]))
                mask = list(map(int, parts[8:12]))
                label = mask.index(1)  # one-hot → class index
                self.samples.append((coords, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        coords, label = self.samples[idx]
        coords = np.array(coords, dtype=np.float32)

        # 数据增强: 随机扰动坐标
        if self.augment:
            noise = np.random.normal(0, 5, size=coords.shape)  # 均值0，std=5 像素扰动
            coords = coords + noise.astype(np.float32)

        return torch.tensor(coords, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ========================
# 模型定义 (大模型)
# ========================
class LargeClassifier(nn.Module):
    def __init__(self):
        super(LargeClassifier, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.GELU()

        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.act4 = nn.GELU()

        self.fc5 = nn.Linear(32, 4)  # 输出4类

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.drop3(self.act3(self.bn3(self.fc3(x))))
        x = self.act4(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

# ========================
# 训练函数
# ========================
def train_model(txt_file, batch_size=64, lr=1e-3, weight_decay=1e-2,
                smoothing=0.1, epochs=30, checkpoint_path="checkpoint.pth"):

    # 数据集
    dataset = RectDataset(txt_file, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LargeClassifier().to(device)

    # 损失函数（Label Smoothing）
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    # 优化器 + 学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1

    # 如果有 checkpoint，加载
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # 记录超参数
    logging.info("===== Training Start =====")
    logging.info(f"Batch size: {batch_size}, LR: {lr}, Optimizer: AdamW, Weight Decay: {weight_decay}")
    logging.info(f"Label smoothing: {smoothing}, Scheduler: CosineAnnealingLR, Epochs: {epochs}")

    # 训练循环
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = 100.0 * correct / total

        # 验证
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= total
        val_acc = 100.0 * correct / total

        # 记录日志
        logging.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # 保存 checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, checkpoint_path)

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix Epoch {epoch}")
        plt.savefig(f"confusion_matrix_epoch{epoch}.png")
        plt.close()

        # 学习率更新
        scheduler.step()

if __name__ == "__main__":
    train_model("rectangles.txt", epochs=30)
