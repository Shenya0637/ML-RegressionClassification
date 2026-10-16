import pandas as pd
import numpy as np

df = pd.read_excel("C:/Users/Lenovo/Desktop/AIP203-assignment1/data/Concrete_Data.xls")

""" print(df.shape)
print(df.columns)
print(df.head())
print(df.info()) """

X = df.iloc[:, :8].to_numpy()
y = df.iloc[:, -1].to_numpy()
""" print(X.shape)
print(y.shape) """

np.random.seed(42)#固定随机种子，随机划分可复现

#随机打乱数据(随机打乱的目的，是避免原始数据排列顺序导致 Development 和 Test 的分布产生不必要的差异。)
indices = np.arange(len(X))
np.random.shuffle(indices)
""" print(indices[:10]) """

#数据划分
test_size = int(len(X) * 0.15)
test_indices = indices[:test_size]
dev_indices = indices[test_size:]

#根据索引取数据
X_test = X[test_indices]
y_test = y[test_indices]
X_dev = X[dev_indices]
y_dev = y[dev_indices]


""" print(X_dev.shape)
print(X_test.shape)
print(y_dev.shape)
print(y_test.shape) """

""" #检查有没有重叠
print("检查重叠索引:")
print(np.intersect1d(test_indices, dev_indices))

#检查有没有重复
print(len(test_indices))
print(len(dev_indices))
print(len(np.unique(test_indices)))
print(len(np.unique(dev_indices))) """

#对dev部分进行随机打乱
dev_indices = indices[test_size:]
np.random.shuffle(dev_indices)
dev_cv_indices = np.arange(len(X_dev))
""" print(dev_indices[:10])
print(len(dev_indices)) """

#5-fold的切分及标准化
folds = np.array_split(dev_cv_indices, 5)
for fold in folds:
    print(len(fold))


mse_scores = []
rmse_scores = []
for i in range(5):
    val_indices = folds[i]
    train_folds = [folds[j] for j in range(5) if j != i]
    train_indices = np.concatenate(train_folds)
    X_train = X_dev[train_indices]
    X_val = X_dev[val_indices]

    #对训练集进行标准化
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    #对训练集和验证集进行标准化
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std

    print('Fold',i)
    print(len(val_indices))
    print(len(np.concatenate(train_folds)))
    print(mean.shape)
    print(std.shape)
    print(X_train_scaled.shape)
    print(X_val_scaled.shape)
    print(np.std(X_train_scaled, axis=0))
    print(np.mean(X_train_scaled, axis=0))

        # 创建一列全为 1 的数组
    bias_column = np.ones((X_train_scaled.shape[0], 1))

    # 将偏置列添加到训练集最左侧
    A_train = np.hstack((bias_column, X_train_scaled))

    print(bias_column.shape)
    print(A_train.shape)

    y_train = y_dev[train_indices]

    print(y_train.shape)

    #使用最小二乘法求出参数
    theta=np.linalg.lstsq(A_train, y_train, rcond=None)
    print(theta[0].shape)

    #设计验证集A的部分
    bias_column_val = np.ones((X_val_scaled.shape[0], 1))
    A_val = np.hstack((bias_column_val, X_val_scaled))
    print(A_val.shape)

    #预测验证集
    y_pred = A_val @ theta[0]
    print(y_pred.shape)

    #计算mse和rmse
    y_val = y_dev[val_indices]
    mse=np.mean((y_pred - y_val) ** 2)
    print("MSE:", mse)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    #把每一个结果都追加在列表里面
    mse_scores.append(mse)
    rmse_scores.append(rmse)
""" #检查5个fold的验证集索引是否有重叠
all_val_indices = np.concatenate(folds)
print(len(all_val_indices))
print(np.sort(all_val_indices)) """ 

mean_mse = np.mean(mse_scores)
mean_rmse = np.mean(rmse_scores)

print("Mean CV MSE:", mean_mse)
print("Mean CV RMSE:", mean_rmse)


# Calculate mean and standard deviation of the full Development Set
dev_mean = np.mean(X_dev, axis=0)
dev_std = np.std(X_dev, axis=0)
X_dev_scaled = (X_dev - dev_mean) / dev_std
print(dev_mean.shape)
print(dev_std.shape)
print(X_dev_scaled.shape)
#构造A_dev
bias_column = np.ones((X_dev_scaled.shape[0], 1))
A_dev = np.hstack((bias_column, X_dev_scaled))
print(A_dev.shape)
#计算theta_final
theta_final = np.linalg.lstsq(A_dev, y_dev, rcond=None)[0]
print(theta_final.shape)
print(theta_final)

#处理测试集
X_test_scaled = (X_test - dev_mean) / dev_std #不能使用测试集的均值和标准差来标准化测试集
print(X_test_scaled.shape)

#构造A_test
bias_test = np.ones((X_test_scaled.shape[0], 1))
A_test = np.hstack((bias_test, X_test_scaled))
print(A_test.shape)

#预测测试集
y_test_pred = A_test @ theta_final
print(y_test_pred.shape)
print(y_test_pred[:5])

#计算测试集的MSE和RMSE
mse_test = np.mean((y_test_pred - y_test) ** 2)
rmse_test = np.sqrt(mse_test)
print("Test MSE:", mse_test)
print("Test RMSE:", rmse_test)

#计算训练集预测值
y_dev_pred = A_dev @ theta_final
print(y_dev_pred.shape)

#计算训练集的MSE和RMSE
train_mse = np.mean((y_dev_pred - y_dev) ** 2)
train_rmse = np.sqrt(train_mse)

print("Training MSE:", train_mse)
print("Training RMSE:", train_rmse)
""" 
#可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))

# Actual vs Predicted
plt.scatter(y_test, y_test_pred, alpha=0.6)

# Perfect prediction line
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--',
    label='Perfect Prediction'
)

plt.xlabel("Actual Strength (MPa)")
plt.ylabel("Predicted Strength (MPa)")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
 """

import numpy as np

# 假设 X_train_scaled 已经完成标准化
# X_train_scaled.shape = (700, 8)

# Step 1: 设置聚类中心数量
K = 20

# Step 2: 随机选择 K 个训练样本作为初始中心
rng = np.random.default_rng(42)

indices = rng.choice(
    X_train_scaled.shape[0],
    size=K,
    replace=False
)

centers = X_train_scaled[indices]

print("Initial centers:", centers.shape)


max_iter = 100
tol = 1e-4

X_expanded = X_train_scaled[:, np.newaxis, :]
for iteration in range(max_iter):
    # 1. Assignment
    centers_expanded = centers[np.newaxis, :, :]

    diff = X_expanded - centers_expanded

    distances_squared = np.sum(diff ** 2, axis=2)

    labels = np.argmin(distances_squared, axis=1)
    # 2. Update
    new_centers = np.zeros_like(centers)
    for j in range(K):
        mask = (labels == j)
        cluster_points = X_train_scaled[mask]
        new_centers[j] = np.mean(cluster_points, axis=0) if np.any(mask) else centers[j]
    # 3. Convergence check
    movement = new_centers - centers

    center_shifts = np.sqrt(
        np.sum(movement ** 2, axis=1)
    )

    max_shift = np.max(center_shifts)

    # 更新 centers
    centers = new_centers

    if max_shift < tol:
        print("Converged at iteration:", iteration + 1)
        break
print("Final centers shape:", centers.shape)
print("Maximum center shift:", max_shift)
print("Iterations completed:", iteration + 1)

# 使用最终的 K-Means centers
diff = X_train_scaled[:, np.newaxis, :] - centers[np.newaxis, :, :]

distances_squared = np.sum(diff ** 2, axis=2)

sigma = 1.0

Phi = np.exp(-distances_squared / (2 * sigma ** 2))

print("Gaussian feature matrix:", Phi.shape)
print("First sample:", Phi[0])