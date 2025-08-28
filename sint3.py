import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.optim import Adam, SGD
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Функция Розенброка
def rosenbrock(X, Y, a=1, b=100):
    return (a - X)**2 + b * (Y - X**2)**2

# Генерация ландшафта
w1_range = np.linspace(-2, 2, 200)
w2_range = np.linspace(-1, 3, 200)
W1, W2 = np.meshgrid(w1_range, w2_range)
loss = rosenbrock(W1, W2)

# Модель
class OptimizationModel(torch.nn.Module):
    def __init__(self, init_w1, init_w2):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor([init_w1], dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor([init_w2], dtype=torch.float32))

    def forward(self):
        a = 1
        b = 100
        return (a - self.w1)**2 + b * (self.w2 - self.w1**2)**2

# Градиентные методы с рестартами
def optimize_with_restarts(optimizer_type='sgd', restarts=5, lr=0.001, n_steps=2000):
    best_result = None
    for _ in range(restarts):
        init_w1 = np.random.uniform(-2, 2)
        init_w2 = np.random.uniform(-1, 3)
        model = OptimizationModel(init_w1, init_w2)

        if optimizer_type == 'sgd':
            optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type == 'adam':
            optimizer = Adam(model.parameters(), lr=lr)

        history = {'w1': [], 'w2': [], 'loss': []}
        for _ in range(n_steps):
            loss_val = model()
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            history['w1'].append(model.w1.item())
            history['w2'].append(model.w2.item())
            history['loss'].append(loss_val.item())

        if best_result is None or history['loss'][-1] < best_result['loss'][-1]:
            best_result = history
    return best_result

# AdaBoost
def optimize_adaboost(n_estimators=100):
    X = np.random.uniform(low=[-2, -1], high=[2, 3], size=(5000, 2))
    y = rosenbrock(X[:, 0], X[:, 1])

    model = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=5),
        n_estimators=n_estimators,
        learning_rate=0.3
    )
    model.fit(X, y)

    grid_x, grid_y = np.mgrid[-2:2:0.05, -1:3:0.05]
    grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    preds = model.predict(grid)
    best_idx = np.argmin(preds)
    best_w1, best_w2 = grid[best_idx]

    return {'w1': [best_w1], 'w2': [best_w2], 'loss': [preds[best_idx]]}

# Оптимизация
hist_sgd = optimize_with_restarts('sgd', restarts=10, lr=0.001, n_steps=2000)
hist_adam = optimize_with_restarts('adam', restarts=10, lr=0.001, n_steps=2000)
hist_adaboost = optimize_adaboost()

# Визуализация
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Поверхность функции
ax.plot_surface(W1, W2, loss, cmap='viridis', alpha=0.2, antialiased=True)

# Функция отрисовки траектории
def plot_trajectory(ax, hist, color, label, final_marker='o', marker_size=80):
    ax.plot(hist['w1'], hist['w2'], hist['loss'],
            color=color, linewidth=3.0, alpha=0.95, label=label)

    ax.scatter(hist['w1'][-1], hist['w2'][-1], hist['loss'][-1],
               c=color, marker=final_marker, s=marker_size, edgecolor='black')

# Траектории
plot_trajectory(ax, hist_sgd, 'red', 'SGD (best restart)', 'o', 80)
plot_trajectory(ax, hist_adam, 'blue', 'Adam (best restart)', 'o', 80)
plot_trajectory(ax, hist_adaboost, 'green', 'AdaBoost', '*', 120)

# Подписи
ax.set_xlabel('Weight 1 (w1)', fontsize=14)
ax.set_ylabel('Weight 2 (w2)', fontsize=14)
ax.set_zlabel('Loss', fontsize=14)
ax.legend(fontsize=14)
plt.title('Оптимизация функции Розенброка: SGD, Adam и AdaBoost', fontsize=16)
plt.tight_layout()
plt.show()

# === Проверка, куда попали оптимизаторы ===
print("SGD final:")
print(f"  w1 = {hist_sgd['w1'][-1]:.4f}, w2 = {hist_sgd['w2'][-1]:.4f}, loss = {hist_sgd['loss'][-1]:.4f}")

print("Adam final:")
print(f"  w1 = {hist_adam['w1'][-1]:.4f}, w2 = {hist_adam['w2'][-1]:.4f}, loss = {hist_adam['loss'][-1]:.4f}")

print("AdaBoost final:")
print(f"  w1 = {hist_adaboost['w1'][0]:.4f}, w2 = {hist_adaboost['w2'][0]:.4f}, loss = {hist_adaboost['loss'][0]:.4f}")