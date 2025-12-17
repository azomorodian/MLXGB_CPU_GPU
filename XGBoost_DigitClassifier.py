import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import xgboost as xgb

# Generate an even larger dataset
np.random.seed(42)
X = np.random.rand(1000000, 100)
y = np.random.rand(1000000)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data: standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GPU version with xgb.train using tree_method='hist' and device='cuda'
params_gpu = {
    'tree_method': 'hist',
    'device': 'cuda',
}

dtrain_gpu = xgb.DMatrix(X_train_scaled, label=y_train)
dtest_gpu = xgb.DMatrix(X_test_scaled, label=y_test)

# Training time on GPU
start_time = time.time()
model_gpu = xgb.train(params_gpu, dtrain_gpu, num_boost_round=20000)
gpu_train_time = time.time() - start_time

# Prediction time on GPU
start_time = time.time()
train_pred_gpu = model_gpu.predict(dtrain_gpu)
test_pred_gpu = model_gpu.predict(dtest_gpu)
gpu_pred_time = time.time() - start_time

# Evaluation
train_mae_gpu = mean_absolute_error(y_train, train_pred_gpu)
test_mae_gpu = mean_absolute_error(y_test, test_pred_gpu)
train_rmse_gpu = mean_squared_error(y_train, train_pred_gpu, squared=False)
test_rmse_gpu = mean_squared_error(y_test, test_pred_gpu, squared=False)

print(f"{'Training Time':<20}: {gpu_train_time:.4f} seconds")
print(f"{'Prediction Time':<20}: {gpu_pred_time:.4f} seconds")
print(f"Train MAE: {train_mae_gpu}, Test MAE: {test_mae_gpu}")
print(f"Train RMSE: {train_rmse_gpu}, Test RMSE: {test_rmse_gpu}")

from xgboost import XGBRegressor
model_cpu = XGBRegressor(n_estimators=20000)

# Training time on CPU
start_time = time.time()
model_cpu.fit(X_train_scaled, y_train)
cpu_train_time = time.time() - start_time

# Prediction time on CPU
start_time = time.time()
train_pred_cpu = model_cpu.predict(X_train_scaled)
test_pred_cpu = model_cpu.predict(X_test_scaled)
cpu_pred_time = time.time() - start_time

# Evaluation
train_mae_cpu = mean_absolute_error(y_train, train_pred_cpu)
test_mae_cpu = mean_absolute_error(y_test, test_pred_cpu)
train_rmse_cpu = mean_squared_error(y_train, train_pred_cpu, squared=False)
test_rmse_cpu = mean_squared_error(y_test, test_pred_cpu, squared=False)

print(f"{'Training Time':<20}: {cpu_train_time:.4f} seconds")
print(f"{'Prediction Time':<20}: {cpu_pred_time:.4f} seconds")
print(f"Train MAE: {train_mae_cpu}, Test MAE: {test_mae_cpu}")
print(f"Train RMSE: {train_rmse_cpu}, Test RMSE: {test_rmse_cpu}")