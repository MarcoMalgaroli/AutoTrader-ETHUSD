import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import matplotlib.pyplot as plt
import copy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_utils.feature_engineering import select_features

with open(Path(__file__).resolve().parent.parent / "config.json", "r") as f:
    CONFIG = json.load(f)

PRINT_WIDTH = CONFIG["print_width"]

# Check for a GPU or use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\x1b[36mUsing device: {device}\x1b[0m")

def set_seed(seed=CONFIG["mlp"]["seed"]):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# --- Configuration ---
BATCH_SIZE = CONFIG["mlp"]["batch_size"]
EPOCHS = CONFIG["mlp"]["epochs"]
LEARNING_RATE = CONFIG["mlp"]["learning_rate"]
HIDDEN_SIZES = CONFIG["mlp"]["hidden_sizes"]
NUM_CLASSES = CONFIG["mlp"]["num_classes"]
DROPOUT = CONFIG["mlp"]["dropout"]


# --- DATA PREPARATION ---
def prepare_dataloader(data: pd.DataFrame, lookahead_days: int = 10, val_pct: float = CONFIG["mlp"]["val_pct"]):
    """
    Prepare dataloaders for training and validation.
    Unlike the LSTM, the MLP uses individual candle feature vectors (no sequences).
    """
    df = data.copy()
    print(f"---> Loading dataset from DataFrame with shape {df.shape}...")
    print(f"  -> Total samples in dataset: {len(df)}")

    valid_data_end = len(df) - lookahead_days
    labeled_df = df.iloc[:valid_data_end].copy()

    split_idx = int(len(labeled_df) * (1 - val_pct))

    train_df = labeled_df.iloc[:split_idx].copy()
    val_df = labeled_df.iloc[split_idx:].copy()

    # Feature selection: auto or manual
    fs_cfg = CONFIG["mlp"]["feature_selection"]
    if fs_cfg["mode"] == "auto":
        feature_cols = select_features(train_df, corr_threshold=fs_cfg["corr_threshold"])
        print(f"  -> Auto-selected features ({len(feature_cols)}): {feature_cols}")
    else:
        feature_cols = fs_cfg["feature_cols"]
        print(f"  -> Manual features ({len(feature_cols)}): {feature_cols}")

    train_df['target'] = train_df['target'].map({-1: 2, 0: 0, 1: 1})
    val_df['target'] = val_df['target'].map({-1: 2, 0: 0, 1: 1})

    print(f"  -> Training samples: {len(train_df)}, from {train_df.index[0]} to {train_df.index[-1]}")
    print(f"  -> Validation samples: {len(val_df)}, from {val_df.index[0]} to {val_df.index[-1]}")

    # Scaling only on train set to avoid data leakage
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    # Train
    X_train_scaled = scaler.transform(train_df[feature_cols])
    y_train = train_df['target'].values

    # Validation
    X_val_scaled = scaler.transform(val_df[feature_cols])
    y_val = val_df['target'].values

    print(f"  -> Shape Input Train: {X_train_scaled.shape}")
    print(f"  -> Shape Target Train: {y_train.shape}")
    print(f"  -> Shape Input Val: {X_val_scaled.shape}")
    print(f"  -> Shape Target Val: {y_val.shape}")

    # Conversion to PyTorch Tensors (no sequences — each sample is a flat feature vector)
    train_data = TensorDataset(
        torch.from_numpy(X_train_scaled).float(),
        torch.from_numpy(y_train).long()
    )
    val_data = TensorDataset(
        torch.from_numpy(X_val_scaled).float(),
        torch.from_numpy(y_val).long()
    )

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE)

    print(f"  -> Number of training batches (batch_size={BATCH_SIZE}): {len(train_loader)}")
    print(f"  -> Number of validation batches (batch_size={BATCH_SIZE}): {len(val_loader)}")

    return train_loader, val_loader, feature_cols, scaler, y_train


# --- THE MLP MODEL ---
class CryptoMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=3, dropout_prob=0.3):
        super(CryptoMLP, self).__init__()

        layers = []
        # Input batch normalization
        layers.append(nn.BatchNorm1d(input_size))

        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_size = h_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_mlp_model(df: pd.DataFrame, lookahead_days=10, plot_results=True):
    print("\n" + " MLP TRAINING ".center(PRINT_WIDTH, "="))

    set_seed(CONFIG["mlp"]["seed"])

    # --- DATA PREPARATION ---
    train_loader, val_loader, feature_cols, scaler, y_train = prepare_dataloader(
        df, lookahead_days=lookahead_days, val_pct=CONFIG["mlp"]["val_pct"]
    )

    # Model Initialization
    model = CryptoMLP(
        len(feature_cols), HIDDEN_SIZES,
        output_size=NUM_CLASSES, dropout_prob=DROPOUT
    ).to(device)

    print(f"\n  -> Model Architecture:\n{model}")

    # Loss and Optimizer
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"  -> Calculated Class Weights (balanced): {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler, plot_results=plot_results)
    return model, scaler, feature_cols


def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, plot_results=True):
    print("\n  -> Starting Training")

    best_loss = float('inf')
    best_val_acc = 0.0
    patience = CONFIG["mlp"]["early_stopping_patience"]
    trigger_times = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on validation set
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total if total > 0 else 0

        print(f"  -> Epoch: {epoch+1:03d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")

        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger_times = 0
            print(" | * New best model")
        else:
            trigger_times += 1
            print(f" | Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"\n  -> Early stopping! Best Validation Loss was: {best_loss:.4f}. Best Accuracy was: {best_val_acc:.4f}")
                break

        if scheduler:
            scheduler.step(avg_val_loss)

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot loss over epochs
    if plot_results:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MLP — Training Loss Over Epochs')
        plt.legend()
        plt.show()


def evaluate(model, test_loader):
    print("\n  -> Evaluating on Test Set")
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    print(f"  -> Accuracy Test Set: {acc:.4f}")
    print("\n  -> Detailed Report:")
    print(classification_report(all_targets, all_preds, labels=[0, 1, 2], target_names=['Hold', 'Long', 'Short'], zero_division=0))
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    print("  -> Confusion Matrix:")
    print(cm)

    # --- TRADING ANALYSIS ---
    print("\n  -> TRADING SIMULATION (High Confidence Only)")
    probs_np = np.array(all_probs)
    targets_np = np.array(all_targets)

    print("\n  -> Probability Stats:")
    print(f"     Max Confidence detected: {np.max(probs_np):.4f}")
    print(f"     Avg Confidence for winner class: {np.max(probs_np, axis=1).mean():.4f}")
    print()

    threshold = CONFIG["trading"]["threshold"]

    for class_idx, class_name in zip([1, 2], ['Long', 'Short']):
        high_conf_idx = np.where(probs_np[:, class_idx] > threshold)[0]

        if len(high_conf_idx) == 0:
            print(f"     {class_name}: No trade above the threshold {threshold*100}%")
            continue

        correct_trades = (targets_np[high_conf_idx] == class_idx).sum()
        total_trades = len(high_conf_idx)
        win_rate = correct_trades / total_trades

        print(f"     {class_name} (> {threshold*100}% conf): {correct_trades}/{total_trades} vinti -> Win Rate: {win_rate:.2%}")

    return (
        pd.Series(all_targets).map({2: -1, 0: 0, 1: 1}),
        pd.Series(all_preds).map({2: -1, 0: 0, 1: 1}),
        probs_np[:, [2, 0, 1]]
    )

def batch_predict_mlp(model, df, feature_cols, scaler, start_idx, end_idx):
    """Execute batch prediction for MLP (single-candle feature vectors)."""
    model.eval()
    valid_indices = list(range(start_idx, end_idx))

    if not valid_indices:
        return [], [], []

    X_batch_raw = df.iloc[valid_indices][feature_cols].copy()
    X_batch_scaled = scaler.transform(X_batch_raw)
    X_batch = torch.FloatTensor(X_batch_scaled).to(device)

    with torch.no_grad():
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    return preds.cpu().numpy(), probs.cpu().numpy(), valid_indices

def predict_next_move(model, df, feature_cols, scaler):
    """
    Predict the action for the next candle based on the most recent candle's features.
    """
    model.eval()

    recent_data = df.iloc[-1:][feature_cols].copy()
    recent_data_scaled = scaler.transform(recent_data)

    input_tensor = torch.tensor(recent_data_scaled).float().to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    # Returns probabilities [Hold, Long, Short]
    return probs.cpu().numpy()[0]
