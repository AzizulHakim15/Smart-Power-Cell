import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from plot_data import plot_predictions, plot_original_vs_downsampled, plot_monthly_splits
#from data_processing import process_all_months
from data_pre_post import process_all_months

# ======================================
# HYPERPARAMETERS
# ======================================
INPUT_STEPS = 500
OUTPUT_STEPS = 500
BATCH_SIZE = 16
EPOCHS = 1
HIDDEN_SIZE1 = 32
HIDDEN_SIZE2 = 64
PATIENCE = 5
LEARNING_RATE = 0.0001
Dropout = 0.1
Dynamic_threshold = 2   # diff between two measured considered same in W
Downsample_size = 50    # downsample steady points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# Load Data
# ======================================
data = pd.read_csv("/home/azizul/LSTM_model/SPC_output_1y.csv")

ref_cols = ['P_I_ref', 'Q_I_ref', 'P_I_ref_int', 'Q_I_ref_int']
weather_cols = ['eta_PV']
time_cols = ['Time']
feature_cols = ref_cols + weather_cols #+ time_cols

meas_cols = ['P_I_meas', 'Q_I_meas','P_I_meas_int','Q_I_meas_int']  # targets

# ======================================
# data processing
# ======================================
#X_train, y_train, X_val, y_val, X_test, y_test , downsampled_data = process_all_months(data, feature_cols, meas_cols, number_of_months=12,dynamic_threshold=Dynamic_threshold,stable_keep_step=Downsample_size)
X_train, y_train, X_val, y_val, X_test, y_test, downsampled_data = process_all_months(
    data,
    feature_cols,
    meas_cols,
    number_of_months=12
)
# ======================================
# SCALING
# ======================================
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(X_train)
y_train_scaled = target_scaler.fit_transform(y_train)

X_val_scaled = feature_scaler.transform(X_val)
y_val_scaled = target_scaler.transform(y_val)

X_test_scaled = feature_scaler.transform(X_test)
y_test_scaled = target_scaler.transform(y_test)

print("Scaling complete.")
print("Scaled shapes -> Train:", X_train_scaled.shape, "Val:", X_val_scaled.shape, "Test:", X_test_scaled.shape)

# ======================================
# Dataset class
# ======================================
class SequenceDataset(Dataset):
    def __init__(self, X, y, input_steps, output_steps):
        self.X = X
        self.y = y
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.indices = list(range(0, len(X) - input_steps - output_steps + 1, 1))
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        X_seq = self.X[start_idx : start_idx + self.input_steps]
        y_seq = self.y[start_idx + self.input_steps :
                       start_idx + self.input_steps + self.output_steps]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

train_dataset = SequenceDataset(X_train_scaled, y_train_scaled, INPUT_STEPS, OUTPUT_STEPS)
val_dataset = SequenceDataset(X_val_scaled, y_val_scaled, INPUT_STEPS, OUTPUT_STEPS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================
# Model
# ======================================
class SpcLSTM(nn.Module):
    def __init__(self, n_features, n_targets):
        super(SpcLSTM, self).__init__()
        self.lstm1 = nn.LSTM(n_features, HIDDEN_SIZE1, batch_first=True)
        self.lstm2 = nn.LSTM(HIDDEN_SIZE1, HIDDEN_SIZE2, batch_first=True)
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(HIDDEN_SIZE2, n_targets)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x

n_features = X_train_scaled.shape[1]
n_targets = y_train_scaled.shape[1]
model = SpcLSTM(n_features, n_targets).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ======================================
# Train
# ======================================
best_val_loss = np.inf
counter = 0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_out = model(X_batch)
        y_out_slice = y_out[:, -OUTPUT_STEPS:, :]
        loss = criterion(y_out_slice, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_out = model(X_batch)
            y_out_slice = y_out[:, -OUTPUT_STEPS:, :]
            val_loss += criterion(y_out_slice, y_batch).item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "dynamic_response_lstm.pth")
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print("Early stopping triggered")
            break
    
# ======================================
# Test function
# ======================================
def test_function(model, X_test, y_test, target_scaler, input_len, output_len, device, start_idx, total_steps):
    model.eval()
    window = X_test[start_idx:start_idx + input_len].copy()
    preds = []

    for t in range(total_steps):
        x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(x_tensor)[:, -output_len, :].cpu().numpy().squeeze()
        preds.append(y_pred)
        next_start = start_idx + t + output_len
        window = X_test[next_start : next_start + input_len].copy()

    pred_array = np.array(preds)
    pred_inv = target_scaler.inverse_transform(pred_array)

    true_start = start_idx + input_len
    true_end = true_start + total_steps
    true_array = y_test[true_start:true_end]
    true_inv = target_scaler.inverse_transform(true_array)

    return true_inv, pred_inv

# ======================================
# Compute metrics
# ======================================
def compute_metrics(true, pred, target_names):
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / true), axis=0) * 100
    overall_mape = np.mean(mape)

    print("\n=== Overall Metrics ===")
    print(f"MSE : {mse:.4f}, RMSE : {rmse:.4f}, R2 : {r2:.4f}, MAPE : {overall_mape:.2f}%")

    print("\n=== Per Target Metrics ===")
    for i, name in enumerate(target_names):
        print(f"{name}: MAPE={mape[i]:.2f}%")

# ======================================
# Save model
# ======================================
def save_trained_model(model, feature_scaler, target_scaler, save_dir="saved_model3", model_name="spc_lstm"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
    joblib.dump(feature_scaler, os.path.join(save_dir, f"{model_name}_feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(save_dir, f"{model_name}_target_scaler.pkl"))
    print(f"Model and scalers saved to '{save_dir}'")

# ======================================
# Run Test, Metrics, Plot, Save
# ======================================
model.load_state_dict(torch.load("dynamic_response_lstm.pth"))
model.eval()

start_idx = 0
total_steps = 170000

true_inv, pred_inv = test_function(
    model, X_test_scaled, y_test_scaled, target_scaler,
    INPUT_STEPS, OUTPUT_STEPS, device,
    start_idx=start_idx, total_steps=total_steps
)

compute_metrics(true_inv, pred_inv, meas_cols)
save_trained_model(model, feature_scaler, target_scaler)
plot_predictions(true_inv, pred_inv, meas_cols, ["W", "Var", "W", "Var"])
plot_original_vs_downsampled(data, downsampled_data)
plot_monthly_splits(
    downsampled_data,
    feature_cols=feature_cols,
    meas_cols=meas_cols,
    number_of_months=12
)

