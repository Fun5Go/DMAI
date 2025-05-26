import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------- 配置参数 ----------
EPOCHS = 40
BATCH_SIZE = 8
LR = 1e-4
DATA_DIR = 'WF-data/train'

# ---------- 图像增强与预处理 ----------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------- 加载数据集 ----------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print("类别映射:", dataset.class_to_idx)
label_counts = Counter([label for _, label in dataset])
print("训练集标签分布:", label_counts)

# ---------- 划分训练/验证集 ----------
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ---------- 设置设备 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 加载预训练模型并微调 ----------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ---------- 定义损失函数和优化器 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ---------- 训练过程 ----------
train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    # ---------- 验证 ----------
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader)
    val_acc = correct / total
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print("预测类别分布：", Counter(all_preds))
    print("实际类别分布：", Counter(all_labels))

# ---------- 保存模型 ----------
torch.save(model.state_dict(), "resnet18_finetuned.pth")
print("模型已保存为 resnet18_finetuned.pth")

# ---------- 绘制曲线 ----------
plt.figure()
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('ResNet18 Accuracy Curve')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve_resnet.png")

plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ResNet18 Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve_resnet.png")
plt.show()
