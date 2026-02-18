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

PRINT_WIDTH = 100

# Check for a GPU or use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\x1b[36mUsing device: {device}\x1b[0m")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ============= H1 =============
# SEQ_LEN = 168 # Number of time steps (candles) to look back
# BATCH_SIZE = 32
# EPOCHS = 100
# LEARNING_RATE = 0.001
# HIDDEN_SIZE = 256
# NUM_LAYERS = 2
# NUM_CLASSES = 3 # 0: Hold, 1: Long, 2: Short
# DROPOUT = 0.5

# ============= D1 =============
SEQ_LEN = 60 # Number of time steps (candles) to look back
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 64
NUM_LAYERS = 3
NUM_CLASSES = 3 # 0: Hold, 1: Long, 2: Short
DROPOUT = 0.3

# --- DATA PREPARATION ---
# Note: each row contains the target of the action to perform at the opening at the next candle
def create_sequences(data, target, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len + 1):
        x = data[i:(i + seq_len)]
        y = target[i + seq_len - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_dataloader(data: pd.DataFrame, lookahead_days: int = 10, val_pct: float = 0.15):
    """
    Prepare dataloaders for training and validation.
    Train goes from 0 to len(df)-predict_window * (1 - val_pct), Validation goes from Train end (including SEQ_LEN-1 candles for context) to end - lookahead_days.
    Args:
        data (pd.DataFrame): The dataset as a pandas DataFrame.
        lookahead_days (int): Number of days that does not have a meaningful label (i.e., the last lookahead_days candles are not used for training).
        val_pct (float): Percentage of the data to use for validation (e.g., 0.15 for 15%).
    """

    df = data.copy()
    df['target'] = df['target'].map({-1: 2, 0: 0, 1: 1}) # Map -1 to 2 (Short), 0 to 0 (Hold), and +1 to 1 (Long)
    print(f"---> Loading dataset from DataFrame with shape {df.shape}...")

    feature_cols = [
        'candle_body', 'candle_shadow_up', 'candle_shadow_low',
        'RSI_15', 'dist_SMA_20', 'dist_SMA_50', 'MACD_norm',
        'ATR_norm', 'BB_width_pct', 'OBV_pct', 'log_ret',
    ]

    print(f"  -> Features ({len(feature_cols)}): {feature_cols}")
    print(f"  -> Total samples in dataset: {len(df)}")

    valid_data_end = len(df) - lookahead_days
    labeled_df = df.iloc[:valid_data_end].copy()
    
    split_idx = int(len(labeled_df) * (1 - val_pct))

    train_df = labeled_df.iloc[:split_idx].copy()
    
    val_df = labeled_df.iloc[split_idx - SEQ_LEN + 1:].copy()
    print(f"  -> Training samples: {len(train_df)}, from {train_df.index[0]} to {train_df.index[-1]}")
    print(f"  -> Validation samples: {len(val_df)}, from {val_df.index[0]} to {val_df.index[-1]} (including {SEQ_LEN-1} candles for context)")

    # Scaling only on train set to avoid data leakage
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    # Train
    X_train_scaled = scaler.transform(train_df[feature_cols])
    y_train = train_df['target'].values
    
    # Validation
    X_val_scaled = scaler.transform(val_df[feature_cols])
    y_val = val_df['target'].values

    # Creating sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, SEQ_LEN)
    print(f"  -> Shape Input Train (number of sequences composed by {SEQ_LEN} time steps): {X_train_seq.shape}")
    print(f"  -> Shape Target Train:  {y_train_seq.shape}")
    print(f"  -> Shape Input Val (number of sequences composed by {SEQ_LEN} time steps): {X_val_seq.shape}")
    print(f"  -> Shape Target Val:  {y_val_seq.shape}")

    # Conversion to PyTorch Tensors
    train_data = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).long())
    print(f"  -> Number of training samples (each composed by {SEQ_LEN} time steps): {len(train_data)}")
    val_data = TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).long())
    print(f"  -> Number of validation samples (each composed by {SEQ_LEN} time steps): {len(val_data)}")
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    print(f"  -> Number of batches in train set (each composed by {BATCH_SIZE} samples): {len(train_loader)}")
    val_loader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE)
    print(f"  -> Number of batches in val set (each composed by {BATCH_SIZE} samples): {len(val_loader)}")

    return train_loader, val_loader, feature_cols, scaler, y_train


# --- THE LSTM MODEL ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = 3, dropout_prob=0.3):
        super(CryptoLSTM, self).__init__()        
        # LSTM Layer
        # batch_first=True means input shape is (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        # Final Fully Connected Layer
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)
        
        # Take only the output of the last time step (the last candle)
        out = out[:, -1, :]
        out = self.dropout(out)
        
        # Pass to the linear layer
        out = self.fc(out)
        return out


def train_lstm_model(df: pd.DataFrame, lookahead_days=10, plot_results=True):
    print("\n" + " LSTM TRAINING ".center(PRINT_WIDTH, "="))
    
    set_seed(42)
    
    # --- DATA PREPARATION ---
    train_loader, val_loader, feature_cols, scaler, y_train = prepare_dataloader(df, lookahead_days=lookahead_days, val_pct=0.15)

    # Model Initialization
    model = CryptoLSTM(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS, output_size = NUM_CLASSES, dropout_prob=DROPOUT).to(device)

    # Loss and Optimizer
    # 1. Calculate weights based on the frequency in the train set
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\n  -> Calculated Class Weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights) # Cross Entropy Loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=1e-5)

    train(model, train_loader, val_loader, criterion, optimizer, plot_results=plot_results)
    return model, scaler, feature_cols


def train(model, train_loader, val_loader, criterion, optimizer, plot_results=True):
    print("\n  -> Starting Training")

    best_loss = float('inf')
    best_val_acc = 0.0
    patience = 15
    trigger_times = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # 1. Zero the old gradients
            
            outputs = model(inputs) # 2. Forward pass (Prediction)
            
            loss = criterion(outputs, labels) # 3. Calculate the error
            
            loss.backward() # 4. Backward pass (Calculate gradients)
            
            optimizer.step() # 5. Optimization (Update weights)
            
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

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total if total > 0 else 0

        print(f"  -> Epoch: {epoch+1:03d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")

        # --- EARLY STOPPING LOGIC ---
        # if avg_val_loss < best_loss:
        if best_val_acc == 0 or val_acc > best_val_acc:
            best_loss = avg_val_loss
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger_times = 0 # Reset patience
            print(" | * New best model")
        else:
            trigger_times += 1
            print(f" | Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"\n  -> Early stopping! Best Validation Loss was: {best_loss:.4f}. Best Accuracy was: {best_val_acc:.4f}")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot loss over epochs
    if plot_results:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
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
            # Get the predicted class (the one with the highest score)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    
    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    print(f"  -> Accuracy Test Set: {acc:.4f}")
    print("\n  -> Detailed Report:")
    print(classification_report(all_targets, all_preds, labels = [0, 1, 2], target_names = ['Hold', 'Long', 'Short'], zero_division=0))
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels = [0, 1, 2])
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

    threshold = 0.40
    
    for class_idx, class_name in zip([1, 2], ['Long', 'Short']):
        # select only the predictions where the probability for this class is > 60%
        high_conf_idx = np.where(probs_np[:, class_idx] > threshold)[0]
        
        if len(high_conf_idx) == 0:
            print(f"     {class_name}: No trade above the threshold {threshold*100}%")
            continue
            
        correct_trades = (targets_np[high_conf_idx] == class_idx).sum()
        total_trades = len(high_conf_idx)
        win_rate = correct_trades / total_trades
        
        print(f"     {class_name} (> {threshold*100}% conf): {correct_trades}/{total_trades} vinti -> Win Rate: {win_rate:.2%}")
    
    return pd.Series(all_targets).map({2: -1, 0: 0, 1: 1}), pd.Series(all_preds).map({2: -1, 0: 0, 1: 1}), probs_np[:, [2, 0, 1]]


def predict_next_move(model, df, feature_cols, scaler):
    """
    Predict the action for the next candle based on the last SEQ_LEN candles.
    """
    model.eval()
    
    # Take last SEQ_LEN rows to create the only important sequence: the one that will be used to predict the next move
    recent_data = df.iloc[-SEQ_LEN:][feature_cols].copy()
    if len(recent_data) < SEQ_LEN:
        raise ValueError(f"Not enough data for prediction. Needed {SEQ_LEN} candles, got {len(recent_data)}")
    
    # Scale the features using the same scaler fitted on the training data
    recent_data_scaled = scaler.transform(recent_data)
    
    # Crea il tensore (Batch=1, Seq=60, Features=...)
    input_tensor = torch.tensor(recent_data_scaled).unsqueeze(0).float().to(device)    
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        
    # Restituisce le probabilità [Hold, Long, Short]
    return probs.cpu().numpy()[0]