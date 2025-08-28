import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка датасета
import kagglehub
path = kagglehub.dataset_download("zarajamshaid/language-identification-datasst")
print("Path to dataset files:", path)

# Чтение CSV
import pandas as pd
data = pd.read_csv(path + "/dataset.csv", sep=',')
print("Пропущенные значения:", data.columns[data.isna().any()].tolist())

# Кодируем языки
data["language"] = data["language"].astype('category').cat.codes
language_count = data["language"].nunique()

# 1. collections.Counter
from collections import Counter

def yield_tokens(data_train):
    for text in data_train:
        yield list(text)

all_tokens = [token for tokens in yield_tokens(data["Text"]) for token in tokens]
counter = Counter(all_tokens)
specials = ["<unk>"]
vocab = {token: idx for idx, token in enumerate(specials + list(counter.keys()))}
default_index = vocab["<unk>"]

# Разделение данных
from sklearn.model_selection import train_test_split

y = data["language"]
X = data["Text"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Класс Dataset
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_text, labels):
        self.data = data_text.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_indices = tokens_to_indices(list(self.data[idx]))
        text_tensor = torch.tensor(token_indices, dtype=torch.int64)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.int64)
        return text_tensor, label_tensor

trainset = TextDataset(X_train, y_train)
testset = TextDataset(X_test, y_test)

# Collate-функция
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for text, label in batch:
        label_list.append(label)
        text_list.append(text)
        offsets.append(text.size(0))
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return label_list, text_list, offsets

# DataLoader
from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=collate_batch)
testloader = DataLoader(testset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# Модель
import torch.nn as nn

vocab_size = len(vocab)
num_class = language_count

class TextClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, 64)  # без sparse=True
        self.fc = nn.Linear(64, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

net = TextClassificationModel()

def tokens_to_indices(tokens):
    return [vocab.get(token, default_index) for token in tokens]

# Обучение модели
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Обычно для Adam берут меньший lr

losses = []
all_losses = []  # (iteration, epoch, loss)
net.train()

iteration = 0

for epoch in range(10):  # или 50
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for labels, inputs, offsets in trainloader:
        optimizer.zero_grad()
        outputs = net(inputs, offsets)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # фиксируем loss на каждой итерации
        all_losses.append((iteration, epoch, loss.item()))
        iteration += 1

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"[Epoch {epoch + 1}] Loss: {running_loss:.4f}, Accuracy: {running_corrects / total:.4f}")
    losses.append(running_loss)

print("Finished Training")

# Тестирование
net.eval()
running_correct = 0
num_of_tests = 0

with torch.no_grad():
    for labels, inputs, offsets in testloader:
        outputs = net(inputs, offsets)
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        num_of_tests += labels.size(0)

print(f"Test Accuracy: {running_correct / num_of_tests:.4f}")

# Первый график

import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, len(losses) + 1)
loss_values = np.array(losses)

plt.figure(figsize=(10, 6))

# Основная линия — функция потерь
plt.plot(epochs, loss_values, color='blue', linestyle='-', label='Loss')

# Точки градиентного спуска
plt.scatter(epochs, loss_values, color='red', zorder=5, label='Adam')

# Стрелки (градиентный спуск)
for i in range(1, len(epochs)):
    plt.annotate('',
                 xy=(epochs[i], loss_values[i]),
                 xytext=(epochs[i-1], loss_values[i-1]),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Настройки графика
plt.title("Функция потерь и путь градиентного спуска")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Второй график

import matplotlib.pyplot as plt
import numpy as np

# Извлекаем значения
iterations, epochs_list, loss_values = zip(*all_losses)
iterations = np.array(iterations)
loss_values = np.array(loss_values)

plt.figure(figsize=(12, 6))

# Линия лосса
plt.plot(iterations, loss_values, color='blue', linestyle='-', label='Функция потерь (Loss)')

# Точки шагов SGD
plt.scatter(iterations, loss_values, color='red', s=20, label='Шаги Adam (итерации)')

# Стрелки градиентного спуска (опционально)
for i in range(1, len(iterations)):
    plt.annotate('',
                 xy=(iterations[i], loss_values[i]),
                 xytext=(iterations[i - 1], loss_values[i - 1]),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                 annotation_clip=False)

# Оформление
plt.title("График функции потерь и шагов градиентного спуска (по итерациям)")
plt.xlabel("Итерация")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()