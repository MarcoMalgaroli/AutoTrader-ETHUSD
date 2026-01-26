import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# --- CONFIGURATION ---
FILE_NAME = Path("datasets", "final", "ETHUSD_M15_176673.csv")
SEQ_LEN = 60 # Number of time steps (candles) to look back
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 128
NUM_LAYERS = 3

# Check if we have a GPU (CUDA for Nvidia, MPS for Mac M1/M2) or use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. DATA PREPARATION ---
def create_sequences(data, target, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = target[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

print("\n--- Loading and Preparing Data ---")
df = pd.read_csv(FILE_NAME)

# Select features (everything except time, target, and raw prices)
feature_cols = [c for c in df.columns if c not in ['time', 'target', 'open', 'high', 'low', 'close']]
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Scaling (0-1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
y_raw = df['target'].values

# Creating sequences
X, y = create_sequences(X_scaled, y_raw, SEQ_LEN)

# Split Train/Test (Temporal, no shuffle)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Conversion to PyTorch Tensors
# .float() is important because network weights are float32
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE) # Shuffle False for time series is better to keep it, although internal batching can be discussed.
test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

print(f"Shape Input Train: {X_train.shape}")
print(f"Shape Input Test:  {X_test.shape}")

# --- 2. THE LSTM MODEL (The Class) ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = 1):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Final Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid for output between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state and cell state to zero
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Take only the output of the last time step (the last candle)
        out = out[:, -1, :]
        
        # Pass to the linear layer
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Model Initialization
input_dim = X_train.shape[2] # Number of features
model = CryptoLSTM(input_dim, HIDDEN_SIZE, NUM_LAYERS).to(device)

# Loss and Optimizer
criterion = nn.BCELoss() # Binary Cross Entropy (for target 0 or 1)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

# --- 3. TRAINING LOOP ---
print("\n--- Starting Training ---")
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
        loss = criterion(outputs, labels.unsqueeze(1)) # unsqueeze is used to match dimensions
        
        # 4. Backward pass (Calculate gradients)
        loss.backward()
        
        # 5. Optimization (Update weights)
        optimizer.step()
        
        train_loss += loss.item()
    
    # Print every 5 epochs
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")

# --- 4. EVALUATION ---
print("\n--- Final Evaluation ---")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Convert probabilities to 0 or 1
        predicted = (outputs > 0.5).float()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

# Calculate metrics
acc = accuracy_score(all_targets, all_preds)
print(f"Accuracy Test Set: {acc:.4f}")
print("\nDetailed Report:")
print(classification_report(all_targets, all_preds))

# --- 5. SAVING ---
# In PyTorch, save the "state_dict" (weights), not the entire object
torch.save(model.state_dict(), "lstm_pytorch_weights.pth")
print("💾 Model saved as 'lstm_pytorch_weights.pth'")