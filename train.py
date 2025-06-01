import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Simple_CNN.model import SimpleCNN
from support import load_dataset
import matplotlib.pyplot as plt
from collections import Counter
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("./Plots", exist_ok=True)
os.makedirs("./Models", exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4

# ---------- Load dataset ----------
train_dataset, _ = load_dataset()
label_counts = Counter([label for _, label in train_dataset])
print("Training set label distribution:", label_counts)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------- Initialize model ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


train_loss_list, train_acc_list = [], []

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

    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

# ---------- Save model ----------
torch.save(model.state_dict(), "./Models/simplecnn_origin.pth")
print("Model saved as simplecnn_origin.pth")

# ---------- Best accuracy ----------
best_acc = max(train_acc_list)
best_epoch = train_acc_list.index(best_acc) + 1
print(f'Best training accuracy: {best_acc:.4f} at epoch {best_epoch}')

# ---------- Plotting ----------
plt.figure()
plt.plot(train_acc_list, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('No finetune Accuracy Curve')
plt.legend()
plt.grid(True)
plt.savefig("./Plots/accuracy_curve_origin.png")

plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('No finetune Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("./Plots/loss_curve_origin.png")
plt.show()

# ---------- Best accuracy ----------
best_acc = max(train_acc_list)
best_epoch = train_acc_list.index(best_acc) + 1
print(f'Best training accuracy: {best_acc:.4f} at epoch {best_epoch}')
