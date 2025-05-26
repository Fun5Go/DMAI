import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
from support import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------- Parameters ----------
BATCH_SIZE = 4
MODEL_PATH = 'Models/simplecnn_finetuned.pth'

# ---------- Load test dataset ----------
_, test_dataset = load_dataset()
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ---------- Load model ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------- Test evaluation ----------
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
print("Predicted label distribution:", Counter(all_preds))
print("Actual label distribution:", Counter(all_labels))

# ---------- Confusion Matrix ----------
cm = confusion_matrix(all_labels, all_preds)
class_names = test_dataset.classes

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Test Accuracy: {accuracy:.2%})')
plt.tight_layout()
plt.savefig("confusion_matrix_test.png")
plt.show()
