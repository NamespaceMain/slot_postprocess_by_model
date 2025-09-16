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
import numpy as np
import cv2
import os
from tqdm import tqdm
from datetime import datetime
from thop import profile, clever_format

# ========================
# 日志配置
# ========================
# 使用当前日期时间生成文件名
current_time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
base_path = "test_2"
# log_dir = "logs"
os.makedirs(base_path, exist_ok=True)
log_filename = os.path.join(base_path, f"train_{current_time}.log")
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ========================
# 数据集定义
# ========================
class PointsDataset(Dataset):
    def __init__(self, txt_file, augment=True, perturb_ratio=0.002):
        """
        augment: 是否进行随机扰动
        perturb_ratio: 坐标扰动比例 (0~1)，相对于最大坐标896
        """
        self.data = []
        self.augment = augment
        self.perturb_ratio = perturb_ratio
        self.max_coord = 896  # 最大坐标值，用于归一化
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                coords = list(map(float, parts[1:15]))  # 10个float
                mask = list(map(int, parts[15:19]))     # one-hot
                label = mask.index(1)                  # 类别索引
                self.data.append((coords, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords, label = self.data[idx]
        coords = np.array(coords, dtype=np.float32)

        # 数据增强: 坐标微扰
        if self.augment:
            perturb = (np.random.rand(14) * 2 - 1) * self.perturb_ratio * self.max_coord
            # print(f"Perturbation: {perturb}")
            coords = coords + perturb
            coords = np.clip(coords, 0, self.max_coord)  # 确保坐标在有效范围内

        # 归一化到 [0,1]
        coords = coords / self.max_coord

        return torch.tensor(coords, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ===================== 模型定义 =====================
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(14, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)  # 输出4类

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class SmallEnhancedClassifier(nn.Module):
    def __init__(self):
        super(SmallEnhancedClassifier, self).__init__()
        self.fc1 = nn.Linear(14, 24)   # 增加第一层宽度
        self.bn1 = nn.BatchNorm1d(24)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(24, 16)  # 压缩特征
        self.bn2 = nn.BatchNorm1d(16)
        self.act2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(0.2)  # 防过拟合

        self.fc3 = nn.Linear(16, 4)   # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x

class LargeClassifier(nn.Module):
    def __init__(self):
        super(LargeClassifier, self).__init__()
        self.fc1 = nn.Linear(14, 128)
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

        self.fc5 = nn.Linear(32, 4)  # 输出层

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.drop3(self.act3(self.bn3(self.fc3(x))))
        x = self.act4(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

# ===================== 可视化函数 =====================
def visualize_predictions(epoch, inputs, labels, preds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img = np.ones((896, 896, 3), dtype=np.uint8) * 255
    for idx, i in tqdm(enumerate(range(len(inputs))[::1000]), total=len(inputs)//1000, desc=f"save vis, epoch{epoch}"):
        points = np.array(inputs[i].reshape(7, 2) * 896, dtype=np.int32)  # 反归一化
        center_pt = points[0]
        corner_pt = points[1:3]
        box_pts = points[3:]
        # 绘制中心点(绿)和角点(红)
        cv2.circle(img, tuple(center_pt), 5, (0, 255, 0), -1)
        cv2.circle(img, tuple(corner_pt[0]), 5, (0, 0, 255), -1)
        cv2.circle(img, tuple(corner_pt[1]), 5, (0, 0, 255), -1)
        cv2.line(img, tuple(corner_pt[0]), tuple(corner_pt[1]), (0, 0, 255), 1)
        # 绘制框(绿)
        # cv2.polylines(img, [box_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for p in box_pts:
            cv2.circle(img, tuple(p), 3, (0, 255, 0), -1)
            cv2.line(img, tuple(p), tuple(center_pt), (0, 255, 0), 1)
        # 真值(紫)
        true_idx = labels[i]
        cv2.circle(img, tuple(box_pts[true_idx]), 5, (125, 0, 125), -1)

        # 预测(蓝)
        pred_idx = preds[i]
        cv2.circle(img, tuple(box_pts[pred_idx]), 8, (255, 0, 0), 2)
        
        if (idx+1)%5 == 0:
            cv2.imwrite(os.path.join(save_dir, f"epoch{epoch}_sample{i}.png"), img)
            img = np.ones((896, 896, 3), dtype=np.uint8) * 255
            
def compute_geo_loss(inputs, outputs, labels):
    """
    inputs: [B, 8] (x1,y1,x2,y2,x3,y3,x4,y4)
    labels: [B] (0~3) 表示哪一个点被替换
    """
    batch_size = inputs.size(0)
    # geo_losses = []
    geo_losses = 0.0
    preds = outputs.argmax(1)
    preds_score = outputs.softmax(1)
    for i in range(batch_size):
        gt_idx = labels[i].item()   # 被替换点索引
        corners = inputs[i].view(7, 2)[1:3]   # [4,2]
        coords = inputs[i].view(7, 2)[3:]   # [4,2]
        corner_1_dis = torch.norm(corners[0] - coords, dim=1)
        corner_2_dis = torch.norm(corners[1] - coords, dim=1)
        corner_idx_0 = corner_1_dis.argmin(0) if corner_1_dis.min() < 10.0/896 else -1
        corner_idx_1 = corner_2_dis.argmin(0) if corner_2_dis.min() < 10.0/896 else -1
        corner_idx_0 = corner_idx_0 if corner_idx_0 != -1 else corner_idx_1
        corner_idx_1 = corner_idx_1 if corner_idx_1 != -1 else corner_idx_0
        # corner_idx_1 = torch.norm(corners[1] - coords, dim=1).argmin(0)
        geo_losses += 0.5*(preds_score[i][corner_idx_0] + preds_score[i][corner_idx_1]) * 100.0
                
        # target_idx = preds[i].item()   # 被替换点索引
        # target_point = coords[target_idx]   # (2,)
        
        # # 计算与其余 3 个点的 L2 距离
        # dists = torch.norm(corners - target_point, dim=1)  # [4]
        # # dists = torch.cat([dists[:target_idx], dists[target_idx+1:]])  # 去掉自身
        # d_min = torch.min(dists)  # 最小距离
        # if d_min < 10.0/896:   
        #     # 定义loss：d 越小，loss 越大
        #     # geo_losses.append(0.1 / (d_min + 1e-6))
        #     geo_losses.append(preds_score[i][target_idx]* 100.0)
        # else:
        #     geo_losses.append(torch.tensor(0.0).to(inputs.device))

    # return torch.mean(torch.stack(geo_losses))
    return geo_losses/batch_size

# ===================== 训练代码（带Checkpoint） =====================
def train(txt_file, epochs=10, batch_size=16, lr=0.001, train_ratio=0.8,
          resume=False, checkpoint_path=None, weight_decay=1e-4, step_size=3, gamma=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 训练集使用数据增强，验证集不增强
    full_dataset = PointsDataset(txt_file)
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.augment = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SmallEnhancedClassifier().to(device)
    # model = LargeClassifier().to(device)
    logging.info(f"Model: {model}\nDevice: {device}")
    
    smoothing = 0.1  # 平滑系数 0~1
    criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    lambda_geo = 0.3  # 几何损失权重

    # 统计 FLOPs 和 参数量
    # 生成一个 dummy 输入，假设 batch_size=1
    dummy_input = torch.randn(1, 14).to(device)   # 输入 7 个 float (7 对点坐标)
    flops, params = profile(model, inputs=(dummy_input, ), verbose=True)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Total FLOPs: {flops}, Total Params: {params}")
    logging.info(f"Total FLOPs: {flops}, Total Params: {params}")
        
    start_epoch = 1

    # ====== 加载 checkpoint ======
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从第 {start_epoch} 个 epoch 继续训练")

    os.makedirs(os.path.join(base_path, "checkpoints"), exist_ok=True)

    # 记录超参数
    logging.info("===== Training Start =====")
    logging.info(f"DATA len: {len(full_dataset)}, Epoch len: {epochs}, Batch size: {batch_size}, LR: {lr}, Optimizer: AdamW, Weight Decay: {weight_decay}")
    logging.info(f"Label smoothing: {smoothing}, Scheduler: StepLR, Epochs: {epochs}, Lambda_geo: {lambda_geo}")
    
    for epoch in range(start_epoch, epochs + 1):
        # ===== 训练阶段 =====
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        ce_loss_total, geo_loss_total = 0.0, 0.0
        batch_ce_loss, batch_geo_loss, batch_loss = 0.0, 0.0, 0.0
        iter_interval = 10  # 每10个batch打印一次
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Train Epoch {epoch}/{epochs}")
        for iter_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # lambda_geo = 0.5  # 几何损失权重
            ce_loss = criterion(outputs, labels) * 10 * (1 - lambda_geo)  # 交叉熵损失
            geo_loss = compute_geo_loss(inputs, outputs, labels) * lambda_geo      # 几何损失
            # print(f"ce_loss: {ce_loss.item()}, geo_loss: {geo_loss.item()}")
            loss = ce_loss + geo_loss           # 总损失
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
            batch_ce_loss += ce_loss.item()
            batch_geo_loss += geo_loss.item()
            
            ce_loss_total += ce_loss.item()
            geo_loss_total += geo_loss.item()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)
            
            if (iter_idx + 1) % iter_interval == 0:
                batch_loss = batch_loss / iter_interval
                batch_ce_loss = batch_ce_loss / iter_interval
                batch_geo_loss = batch_geo_loss / iter_interval
                pbar.set_postfix({"Train Loss": batch_loss,
                "ce_loss": batch_ce_loss, 
                "geo_loss": batch_geo_loss})
                batch_ce_loss, batch_geo_loss, batch_loss = 0.0, 0.0, 0.0
        scheduler.step()  # 更新学习率

        # ===== 验证阶段 =====
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        all_inputs_train, all_labels_train, all_preds_train = [], [], []
        all_inputs_val, all_labels_val, all_preds_val = [], [], []

        with torch.no_grad():
            # 训练集可视化数据
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(1)

                all_inputs_train.append(inputs.cpu().numpy())
                all_labels_train.append(labels.cpu().numpy())
                all_preds_train.append(preds.cpu().numpy())

            # 验证集损失和可视化数据
            for inputs, labels in tqdm(val_loader, total=len(val_loader), desc=f"Val Epoch {epoch}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                total_val += labels.size(0)

                preds = outputs.argmax(1)
                all_inputs_val.append(inputs.cpu().numpy())
                all_labels_val.append(labels.cpu().numpy())
                all_preds_val.append(preds.cpu().numpy())

        train_acc = train_correct / total_train
        val_acc = val_correct / total_val

        # 记录日志
        logging.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss/ len(train_loader):.4f}, "
            f"ce_loss: {ce_loss_total/ len(train_loader):.2f}, geo_loss: {geo_loss_total/ len(train_loader):.2f}, "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%"
        )
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"ce_loss: {ce_loss_total/ len(train_loader):.2f}, geo_loss: {geo_loss_total/ len(train_loader):.2f}, "
              f"Acc: {train_acc:.2f} "
              f"| Val Loss: {val_loss / len(val_loader):.4f} Acc: {val_acc:.2f}")

        # 保存可视化
        all_inputs_train = np.vstack(all_inputs_train)
        all_labels_train = np.hstack(all_labels_train)
        all_preds_train = np.hstack(all_preds_train)
        visualize_predictions(epoch, all_inputs_train, all_labels_train, all_preds_train, os.path.join(base_path,"vis_results/train"))

        all_inputs_val = np.vstack(all_inputs_val)
        all_labels_val = np.hstack(all_labels_val)
        all_preds_val = np.hstack(all_preds_val)
        visualize_predictions(epoch, all_inputs_val, all_labels_val, all_preds_val, os.path.join(base_path,"vis_results/val"))

        # 保存 checkpoint
        if epoch%5==0:
            checkpoint_file = os.path.join(base_path, f"checkpoints/checkpoint_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader)
            }, checkpoint_file)
            print(f"Checkpoint 已保存: {checkpoint_file}")

        # 混淆矩阵
        cm = confusion_matrix(all_labels_val, all_preds_val, labels=[0,1,2,3])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix Epoch {epoch}")
        os.makedirs(os.path.join(base_path, "confusion_matrix"), exist_ok=True)
        plt.savefig(os.path.join(base_path, f"confusion_matrix/confusion_matrix_epoch{epoch}.png"))
        plt.close()

    print("训练完成！")

# ===================== 运行训练 =====================
if __name__ == "__main__":
    # 训练模式
    train("rectangles_1.txt", epochs=30, batch_size=8, lr=0.001,
          resume=False, checkpoint_path="checkpoints/checkpoint_epoch20.pth")

    # 继续训练模式（示例）
    # train("rectangles.txt", epochs=10, batch_size=8, lr=0.001,
    #       resume=True, checkpoint_path="checkpoints/checkpoint_epoch5.pth")
