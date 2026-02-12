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

SEQ_LEN = 20 # Number of time steps (candles) to look back
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 16
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

def prepare_dataloader(file_name):
    """
    Prepare dataloaders for training, validation, and testing from the dataset located at file_name.
    Args:
        file_name (str): Path to the CSV file containing the dataset.
    """
    print(f"---> Loading dataset from {file_name}...")

    df = pd.read_csv(file_name)

    # Select features (everything except time, target, and raw prices)
    # cols_to_drop = [
    #     'time', 'target', 'open', 'high', 'low', 'close', 'tick_volume', 'spread',
    #     'SMA_5', 'EMA_5', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50',
    #     'ATR_14', 'MACD', 'vol_SMA_20',
    #     'RSI_5', 'RSI_10', 'RSI_20', # keep only RSI 15
    #     'dist_SMA_5', 'dist_EMA_5',
    #     'dist_SMA_10', 'dist_EMA_10',
    #     'dist_EMA_20', 'dist_EMA_50',
    #     'KC_width_pct', 'KC_pct'
    # ]
    # feature_cols = [c for c in df.columns if c not in cols_to_drop]

    feature_cols = [
        'RSI_15',          # Momentum classico
        'dist_SMA_20',     # Mean Reversion (distanza dalla media)
        'dist_SMA_50',     # Trend di medio termine
        'MACD_norm',       # Trend momentum
        'ATR_norm',        # Volatilità
        'BB_width_pct',    # Compressione/Esplosione volatilità
        'OBV_pct',         # Pressione volumetrica
        'log_ret',         # Rendimento logaritmico (il movimento puro)
    ]

    print(f"  -> Features ({len(feature_cols)}): {feature_cols}")
    print(f"  -> Total samples in dataset: {len(df)}")
    df['target'] = df['target'].map({-1: 2, 0: 0, 1: 1}) # Map -1 to 2 (Short), 0 to 0 (Hold), and +1 to 1 (Long)

    n = len(df)
    train_end = int(n * 0.85)
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end - SEQ_LEN + 1:].copy()
    print(f"  -> Training samples: {len(train_df)}")
    print(f"  -> Testing samples: {len(test_df)}")

    # Scaling only on train set to avoid data leakage
    scaler = RobustScaler()
    # Train
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    y_train_raw = train_df['target'].values
    # Test
    X_test_scaled = scaler.transform(test_df[feature_cols])
    y_test_raw = test_df['target'].values

    # Creating sequences
    X_train, y_train = create_sequences(X_train_scaled, y_train_raw, SEQ_LEN)
    X_test, y_test = create_sequences(X_test_scaled, y_test_raw, SEQ_LEN)
    print(f"  -> Shape Input Train (number of sequences composed by {SEQ_LEN} time steps): {X_train.shape}")
    print(f"  -> Shape Target Train:  {y_train.shape}")
    print(f"  -> Shape Input Test (number of sequences composed by {SEQ_LEN} time steps): {X_test.shape}")
    print(f"  -> Shape Target Test:  {y_test.shape}")

    # Conversion to PyTorch Tensors
    train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    print(f"  -> Number of training samples (each composed by {SEQ_LEN} time steps): {len(train_data)}")
    print(f"  -> Number of testing samples (each composed by {SEQ_LEN} time steps): {len(test_data)}")
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    print(f"  -> Number of batches in train set (each composed by {BATCH_SIZE} samples): {len(train_loader)}")
    print(f"  -> Number of batches in test set (each composed by {BATCH_SIZE} samples): {len(test_loader)}")

    scaler_path = Path('models/saved/scaler.gz')
    joblib.dump(scaler, scaler_path)
    print(f"\x1b[32;1mScaler saved to {scaler_path}\x1b[0m")

    return train_loader, test_loader, feature_cols, y_train


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


def train_lstm_model(file_name):
    print("\n" + " LSTM TRAINING ".center(PRINT_WIDTH, "="))
    # --- DATA PREPARATION ---
    train_loader, test_loader, feature_cols, y_train = prepare_dataloader(file_name)

    # Model Initialization
    model = CryptoLSTM(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS, output_size = NUM_CLASSES, dropout_prob=DROPOUT).to(device)

    # Loss and Optimizer
    # 1. Calculate weights based on the frequency in the train set
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\n  -> Calculated Class Weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights) # Cross Entropy Loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=1e-5)

    train(model, train_loader, test_loader, criterion, optimizer)
    targets, preds = evaluate(model, test_loader)
    return model, targets, preds


def train(model, train_loader, test_loader, criterion, optimizer):
    print("\n  -> Starting Training")

    best_acc = 0.0
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

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total

        print(f"  -> Epoch: {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")

        # --- EARLY STOPPING LOGIC ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger_times = 0 # Reset pazienza
            print(" | * New best model")
        else:
            trigger_times += 1
            print(f" | Trigger Times: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print(f"\n  -> Early stopping! Best Accuracy was: {best_acc:.4f}")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss VS Validation Loss Over Epochs')
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
    print(classification_report(all_targets, all_preds, labels = [0, 1, 2], target_names = ['Hold', 'Long', 'Short']))
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
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
    
    return all_targets, pd.Series(all_preds).map({2: -1, 0: 0, 1: 1})


def predict_last_candle(model_path, scaler_path, df_path, feature_cols):
    """
    Predict the action for the next candle based on the last SEQ_LEN candles.
    """
    model = CryptoLSTM(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(device)
    model.load_state_dict(torch.load(model_path))

    scaler = joblib.load(scaler_path)
    df = pd.read_csv(df_path)
    
    model.eval()

    last_sequence_df = df[feature_cols].tail(SEQ_LEN)
    
    if len(last_sequence_df) < SEQ_LEN:
        print("Error: Not enough candles to create a sequence!")
        return None
    last_sequence_scaled = scaler.transform(last_sequence_df)
    
    input_tensor = torch.from_numpy(last_sequence_scaled).float().unsqueeze(0).to(device)
    
    # Inference      
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    # Map the output (0: Hold, 1: Long, 2: Short)
    mapping = {0: "HOLD", 1: "LONG", 2: "SHORT"}
    
    return {
        "action": mapping[int(predicted_class.item())],
        "confidence": confidence.item(),
        "probabilities": probabilities.cpu().numpy()[0]
    }