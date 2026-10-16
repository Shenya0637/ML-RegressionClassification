import pandas as pd
import numpy as np

df = pd.read_excel("Concrete_Data.xls")

X = df.iloc[:, :8].to_numpy()
y = df.iloc[:, -1].to_numpy()

indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)
train_size = int(len(X) * 0.70)
val_size = int(len(X) * 0.15)
test_size = int(len(X) * 0.15)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_val = X[val_indices]
y_val = y[val_indices]
X_test = X[test_indices]
y_test = y[test_indices]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - train_mean) / (train_std + 1e-8)
X_val_scaled = (X_val - train_mean) / (train_std + 1e-8)
X_test_scaled = (X_test - train_mean) / (train_std + 1e-8)

bias_train = np.ones((len(X_train_scaled), 1))
A_train = np.hstack((bias_train, X_train_scaled))

bias_val = np.ones((len(X_val_scaled), 1))
A_val = np.hstack((bias_val, X_val_scaled))

bias_test = np.ones((len(X_test_scaled), 1))
A_test = np.hstack((bias_test, X_test_scaled))

theta = np.linalg.lstsq(A_train, y_train, rcond=None)[0]
print("Theta shape:", theta.shape)


y_train_pred = A_train @ theta
y_val_pred = A_val @ theta

val_mse = np.mean((y_val - y_val_pred) ** 2)
val_rmse = np.sqrt(val_mse)

y_test_pred = A_test @ theta
test_mse = np.mean((y_test - y_test_pred) ** 2)
test_rmse = np.sqrt(test_mse)

print("Validation MSE:", val_mse)
print("Validation RMSE:", val_rmse)
print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)


K_values = [10, 20, 40]
sigma_values = [0.5, 1.0, 2.0]

def gaussian_features(X, centers, sigma):
    result = []

    for i in range(len(X)):
        x = X[i]
        features = []

        for j in range(len(centers)):
            mu = centers[j]
            distance_squared = np.sum((x - mu) ** 2)
            phi = np.exp(-distance_squared / (2 * sigma ** 2))
            features.append(phi)

        result.append(features)

    return np.array(result)

max_K = max(K_values)
center_indices = np.random.choice(
    len(X_train_scaled),
    size=max_K,
    replace=False
)

best_rmse = float("inf")
best_K = None
best_sigma = None

for K in K_values:
    centers = X_train_scaled[center_indices[:K]]
    for sigma in sigma_values:
        gaussian_train = gaussian_features(X_train_scaled, centers, sigma)
        gaussian_val = gaussian_features(X_val_scaled, centers, sigma)

        bias_gaussian_train = np.ones((len(gaussian_train), 1))
        G_train = np.hstack((bias_gaussian_train, gaussian_train))

        bias_gaussian_val=np.ones((len(gaussian_val), 1))
        G_val=np.hstack((bias_gaussian_val, gaussian_val))

        gaussian_theta = np.linalg.lstsq(G_train, y_train, rcond=None)[0]
        gaussian_val_pred = G_val @ gaussian_theta
        gaussian_val_mse = np.mean((y_val - gaussian_val_pred) ** 2)
        gaussian_val_rmse = np.sqrt(gaussian_val_mse)
        print(f"K={K}, sigma={sigma}: Gaussian Validation RMSE: {gaussian_val_rmse}")

        if gaussian_val_rmse < best_rmse:
            best_rmse = gaussian_val_rmse
            best_K = K
            best_sigma = sigma
print(f"Best K: {best_K}, Best sigma: {best_sigma}, Best Gaussian Validation RMSE: {best_rmse}")
        
final_centers = X_train_scaled[center_indices[:best_K]]
final_gaussian_train = gaussian_features(X_train_scaled, final_centers, best_sigma)
final_gaussian_test = gaussian_features(X_test_scaled, final_centers, best_sigma)

final_G_train = np.hstack((
    np.ones((len(final_gaussian_train), 1)),
    final_gaussian_train
))
final_G_test = np.hstack((
    np.ones((len(final_gaussian_test), 1)),
    final_gaussian_test
))
final_theta = np.linalg.lstsq(final_G_train, y_train, rcond=None)[0]
final_test_pred = final_G_test @ final_theta
final_test_mse = np.mean((y_test - final_test_pred) ** 2)
final_test_rmse = np.sqrt(final_test_mse)
print("Final Gaussian Theta Shape:", final_theta.shape)
print("Final Gaussian Test RMSE:", final_test_rmse)
