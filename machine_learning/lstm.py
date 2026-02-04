import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import matplotlib.pyplot as plt
import copy

# Check for a GPU or use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- CONFIGURATION ---
# FILE_NAME = Path("datasets", "final", "ETHUSD_D1_3059.csv")
FILE_NAME = Path("datasets", "allIndicators", "ETHUSD_H1_48306.csv")
# FILE_NAME = Path("datasets", "final", "ETHUSD_M15_176764.csv")
# FILE_NAME = Path("datasets", "final", "ETHUSD_M5_518717.csv")

SEQ_LEN = 60 # Number of time steps (candles) to look back
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 3 # 0: Hold, 1: Long, 2: Short
DROPOUT = 0.4


# --- 1. DATA PREPARATION ---
def create_sequences(data, target, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len + 1):
        x = data[i:(i + seq_len)]
        y = target[i + seq_len - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

print("\n--- Loading and Preparing Data ---")
df = pd.read_csv(FILE_NAME)
# Select features (everything except time, target, and raw prices)
cols_to_drop = [
    'time', 'target', 'open', 'high', 'low', 'close',
    'SMA_5', 'EMA_5', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50',
    'ATR_14', 'MACD', 'vol_SMA_20', 'tick_volume'
]
feature_cols = [c for c in df.columns if c not in cols_to_drop]
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Total samples in dataset: {len(df)}")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Testing samples: {len(test_df)}")

# Scaling only on train set to avoid data leakage
scaler = RobustScaler()
# Train
X_train_scaled = scaler.fit_transform(train_df[feature_cols])
y_train_raw = train_df['target'].values
# Validation
X_val_scaled = scaler.transform(val_df[feature_cols])
y_val_raw = val_df['target'].values
# Test
X_test_scaled = scaler.transform(test_df[feature_cols])
y_test_raw = test_df['target'].values

# Creating sequences
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, SEQ_LEN)
X_val, y_val = create_sequences(X_val_scaled, y_val_raw, SEQ_LEN)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, SEQ_LEN)
print(f"Shape Input Train (number of sequences composed by {SEQ_LEN} time steps): {X_train.shape}")
print(f"Shape Target Train:  {y_train.shape}")
print(f"Shape Input Validation (number of sequences composed by {SEQ_LEN} time steps): {X_val.shape}")
print(f"Shape Target Validation:  {y_val.shape}")
print(f"Shape Input Test (number of sequences composed by {SEQ_LEN} time steps): {X_test.shape}")
print(f"Shape Target Test:  {y_test.shape}")

# Conversion to PyTorch Tensors
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
print(f"Number of training samples (each composed by {SEQ_LEN} time steps): {len(train_data)}")
print(f"Number of validation samples (each composed by {SEQ_LEN} time steps): {len(val_data)}")
print(f"Number of testing samples (each composed by {SEQ_LEN} time steps): {len(test_data)}")
train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
print(f"Number of batches in train set (each composed by {BATCH_SIZE} samples): {len(train_loader)}")
print(f"Number of batches in validation set (each composed by {BATCH_SIZE} samples): {len(val_loader)}")
print(f"Number of batches in test set (each composed by {BATCH_SIZE} samples): {len(test_loader)}")

# --- 2. THE LSTM MODEL (The Class) ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = 3, dropout_prob=0.3):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        # Final Fully Connected Layer
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state to zero
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Take only the output of the last time step (the last candle)
        out = out[:, -1, :]
        out = self.dropout(out)
        
        # Pass to the linear layer
        out = self.fc(out)
        return out

# Model Initialization
model = CryptoLSTM(X_train.shape[2], HIDDEN_SIZE, NUM_LAYERS, output_size = NUM_CLASSES, dropout_prob=DROPOUT).to(device)

# Loss and Optimizer
# 1. Calculate weights based on the frequency in the train set
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Calculated Class Weights: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights) # Cross Entropy Loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=1e-4)

# --- 3. TRAINING LOOP ---
print("\n--- Starting Training ---")
patience = 15
best_val_loss = float('inf')
counter = 0
best_model_wts = copy.deepcopy(model.state_dict())
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. Zero the old gradients
        optimizer.zero_grad()
        
        # 2. Forward pass (Prediction)
        outputs = model(inputs)
        
        # 3. Calculate the error
        loss = criterion(outputs, labels)
        
        # 4. Backward pass (Calculate gradients)
        loss.backward()
        
        # 5. Optimization (Update weights)
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", end="")
    
    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0
        print(" * Best Model")
    else:
        counter += 1
        print(f" | Patience: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered!")
            break

# Load the best model weights
model.load_state_dict(best_model_wts)

# Plot loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# --- 4. TEST ---
print("\n--- Final Test ---")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Get the predicted class (the one with the highest score)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

# Calculate metrics
acc = accuracy_score(all_targets, all_preds)
print(f"Accuracy Test Set: {acc:.4f}")
print("\nDetailed Report:")
print(classification_report(all_targets, all_preds, labels = [-1, 0, 1], target_names = ['Hold', 'Long', 'Short']))
# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
print("Confusion Matrix:")
print(cm)
# --- 5. SAVING ---
# In PyTorch, save the "state_dict" (weights), not the entire object
# torch.save(model.state_dict(), "lstm_pytorch_weights.pth")
# print("Model saved as 'lstm_pytorch_weights.pth'")