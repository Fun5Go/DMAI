from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def plot_curves(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, 'r-', label='Train Loss')
    plt.plot(epochs, val_loss, 'b-', label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")

    plt.figure()
    plt.plot(epochs, train_acc, 'r-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'b-', label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve.png")

    plt.show()
