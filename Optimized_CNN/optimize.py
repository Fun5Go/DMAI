import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

import optuna
from optuna.samplers import RandomSampler, TPESampler
import random
import matplotlib.pyplot as plt
import os
import sys
# Add project root to path and set cwd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
print("Current working dir set to:", os.getcwd())

from Simple_CNN.model import SimpleCNN
from support import load_dataset
# ---------- Parameters ----------
BATCH_SIZE = 4
NUM_CLASSES = 2
KFOLD_SPLITS = 5
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load full training dataset ----------
dataset, _ = load_dataset()

# ------------ Save directory for plots ------------
SAVE_DIR_Plots = './Plots/TPE'
os.makedirs(SAVE_DIR_Plots, exist_ok=True)

# ---------- 5-Fold Cross Validation Function ----------
def cross_validate(hparams):
    kfold = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
    val_accuracies = []
    train_loss_curve = []
    val_loss_curve = []
    train_acc_curve = []
    val_acc_curve = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n[Hyperparameter Trial] {hparams} | Fold {fold + 1}/{KFOLD_SPLITS}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=hparams['batch_size'])

        model = SimpleCNN(
            num_filters=hparams['num_filters'],
            dropout=hparams['dropout']
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
        criterion = nn.CrossEntropyLoss()

        fold_train_losses = []
        fold_val_losses = []
        fold_train_acc = []
        fold_val_acc = []


        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            train_correct, train_total = 0, 0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = train_correct / train_total
            fold_train_losses.append(epoch_train_loss)
            fold_train_acc.append(epoch_train_acc)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = val_correct / val_total
            fold_val_losses.append(epoch_val_loss)
            fold_val_acc.append(epoch_val_acc)

            print(f"Epoch {epoch+1:02d}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f} | Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")

            # Early stopping if validation accuracy doesn't change across epochs
            if fold == 0 and len(fold_val_acc) >= 15 and all(val == fold_val_acc[-1] for val in fold_val_acc[-15:]):
                print("Early stopping: validation accuracy has not changed for 15 epochs.")
                return 0.0  # Skip this trial completely  # Skip this trial completely

        train_loss_curve.append(fold_train_losses)
        val_loss_curve.append(fold_val_losses)
        train_acc_curve.append(fold_train_acc)
        val_acc_curve.append(fold_val_acc)

        # Final validation accuracy after all epochs
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_accuracies.append(correct / total)

    avg_accuracy = sum(val_accuracies) / len(val_accuracies)
    print(f"Final avg val accuracy: {avg_accuracy:.4f}")

    param_str = f"lr{hparams['lr']:.0e}_do{hparams['dropout']:.2f}_nf{hparams['num_filters']}_bs{hparams['batch_size']}_nl{hparams['n_layers']}"
    # Plot average loss and accuracy curves across folds
    avg_train_loss = [sum(epoch_vals)/len(epoch_vals) for epoch_vals in zip(*train_loss_curve)]
    avg_val_loss = [sum(epoch_vals)/len(epoch_vals) for epoch_vals in zip(*val_loss_curve)]
    avg_train_acc = [sum(epoch_vals)/len(epoch_vals) for epoch_vals in zip(*train_acc_curve)]
    avg_val_acc = [sum(epoch_vals)/len(epoch_vals) for epoch_vals in zip(*val_acc_curve)]

    plt.figure()
    plt.plot(avg_train_loss, label='Avg Train Loss')
    plt.plot(avg_val_loss, label='Avg Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Average Loss Curve - {param_str}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR_Plots, f"loss_curve_avg_{param_str}.png"))
    plt.close()

    plt.figure()
    plt.plot(avg_train_acc, label='Avg Train Accuracy')
    plt.plot(avg_val_acc, label='Avg Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"Average Accuracy Curve - {param_str}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR_Plots, f"acc_curve_avg_{param_str}.png"))
    plt.close()

    plt.figure()
    for i, (train_acc, val_acc) in enumerate(zip(train_acc_curve, val_acc_curve)):
        plt.plot(train_acc, linestyle='--', label=f'Train Acc Fold {i+1}')
        plt.plot(val_acc, linestyle='-', label=f'Val Acc Fold {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy Curves per Fold - {param_str}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR_Plots, f"acc_curve_folds_{param_str}.png"))
    plt.close()

    return avg_accuracy

# ---------- Optuna Objective Function ----------
def objective(trial):
    hparams = {
        'lr': trial.suggest_categorical('lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]),
        'num_filters': trial.suggest_categorical('num_filters', [32, 64, 128]),
        'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8]),
        'n_layers': trial.suggest_int('n_layers', 2, 3),
    }
    return cross_validate(hparams)

# ---------- Run Optimization ----------
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction='maximize', sampler=TPESampler()) # Bayesian optimization - TPE
study.optimize(objective, n_trials=10)
# study = optuna.create_study(direction='maximize', sampler=RandomSampler()) # Random search
# study.optimize(objective, n_trials=20)
print("Best trial:")
print(study.best_trial)
