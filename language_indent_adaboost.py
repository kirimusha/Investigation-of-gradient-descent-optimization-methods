import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import kagglehub

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Загрузка датасета ---
path = kagglehub.dataset_download("zarajamshaid/language-identification-datasst")
print("Path to dataset files:", path)

data = pd.read_csv(path + "/dataset.csv", sep=',')
print("Пропущенные значения:", data.columns[data.isna().any()].tolist())

# Кодируем языки
data["language"] = data["language"].astype('category').cat.codes
language_count = data["language"].nunique()

# Создание словаря и индексов
def yield_tokens(data_train):
    for text in data_train:
        yield list(text)

all_tokens = [token for tokens in yield_tokens(data["Text"]) for token in tokens]
counter = Counter(all_tokens)
specials = ["<unk>"]
vocab = {token: idx for idx, token in enumerate(specials + list(counter.keys()))}
default_index = vocab["<unk>"]

def tokens_to_indices(tokens):
    return [vocab.get(token, default_index) for token in tokens]

# Разделение данных
y = data["language"]
X = data["Text"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Класс Dataset
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

# DataLoader (shuffle=False для train, важный момент для весов AdaBoost)
trainloader = DataLoader(trainset, batch_size=8, shuffle=False, collate_fn=collate_batch)
testloader = DataLoader(testset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# Модель
vocab_size = len(vocab)
num_class = language_count

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, 64, sparse=False)  # sparse=False здесь
        self.fc = nn.Linear(64, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
class AdaBoostNN:
    def __init__(self, n_estimators, vocab_size, num_class, device):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.vocab_size = vocab_size
        self.num_class = num_class
        self.device = device

    def fit(self, trainloader, testloader):
        n_samples = len(trainloader.dataset)
        sample_weights = torch.ones(n_samples) / n_samples

        criterion = nn.CrossEntropyLoss(reduction='none')
        num_epochs_per_estimator = 5

        train_losses, train_accuracies, test_accuracies = [], [], []

        for m in range(self.n_estimators):
            net = TextClassificationModel(self.vocab_size, self.num_class).to(self.device)
            optimizer = optim.Adam(net.parameters(), lr=0.001)

            for epoch in range(num_epochs_per_estimator):
                net.train()
                idx_sample = 0
                running_loss = 0
                running_correct = 0
                total_samples = 0

                for labels, inputs, offsets in trainloader:
                    optimizer.zero_grad()
                    labels = labels.to(self.device)
                    inputs = inputs.to(self.device)
                    offsets = offsets.to(self.device)

                    outputs = net(inputs, offsets)
                    losses = criterion(outputs, labels)

                    batch_weights = sample_weights[idx_sample: idx_sample + labels.size(0)].to(self.device)
                    batch_weights = batch_weights / batch_weights.sum()  # нормализуем внутри батча

                    loss = (losses * batch_weights).sum()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * labels.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    idx_sample += labels.size(0)

            train_loss = running_loss / total_samples
            train_acc = running_correct / total_samples

            # Подсчёт ошибок для обновления весов
            net.eval()
            incorrect_mask = torch.zeros(n_samples, dtype=torch.bool)
            with torch.no_grad():
                idx_sample = 0
                for labels, inputs, offsets in trainloader:
                    labels = labels.to(self.device)
                    inputs = inputs.to(self.device)
                    offsets = offsets.to(self.device)

                    outputs = net(inputs, offsets)
                    preds = torch.argmax(outputs, dim=1)

                    batch_size = labels.size(0)
                    incorrect_mask[idx_sample:idx_sample + batch_size] = (preds.cpu() != labels.cpu())
                    idx_sample += batch_size

            err_m = torch.sum(sample_weights * incorrect_mask.float()) / torch.sum(sample_weights)
            alpha_m = 0.5 * torch.log((1 - err_m) / (err_m + 1e-10))

            sample_weights = sample_weights * torch.exp(alpha_m * incorrect_mask.float())
            sample_weights = sample_weights / torch.sum(sample_weights)

            self.models.append(net)
            self.alphas.append(alpha_m.item())

            test_acc = self.evaluate(testloader)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            print(f"Estimator {m + 1}/{self.n_estimators} — Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}")

        return train_losses, train_accuracies, test_accuracies

    def evaluate(self, dataloader):
        correct = 0
        total = 0
        for labels, inputs, offsets in dataloader:
            labels = labels.to(self.device)
            inputs = inputs.to(self.device)
            offsets = offsets.to(self.device)
            preds = self.predict_single_batch(inputs, offsets)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / total

    def predict_single_batch(self, inputs, offsets):
        total_preds = None
        with torch.no_grad():
            for alpha, model in zip(self.alphas, self.models):
                outputs = model(inputs, offsets)
                probs = torch.softmax(outputs, dim=1)
                weighted_probs = probs * alpha
                if total_preds is None:
                    total_preds = weighted_probs
                else:
                    total_preds += weighted_probs
        return torch.argmax(total_preds, dim=1)

    def predict(self, dataloader):
        all_preds = []
        with torch.no_grad():
            for labels, inputs, offsets in dataloader:
                inputs = inputs.to(self.device)
                offsets = offsets.to(self.device)
                preds = self.predict_single_batch(inputs, offsets)
                all_preds.append(preds.cpu())
        return torch.cat(all_preds)
    

 # --- Запуск обучения AdaBoost с нейросетью ---

ada = AdaBoostNN(n_estimators=10, vocab_size=vocab_size, num_class=num_class, device=device)

train_losses, train_accs, test_accs = ada.fit(trainloader, testloader)

# Итоговая точность на тесте
preds = ada.predict(testloader)
labels = torch.cat([labels for labels, _, _ in testloader])
accuracy = (preds == labels).float().mean().item()
print(f"Итоговая точность AdaBoost: {accuracy:.4f}")

import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, len(train_losses) + 1)
loss_values = np.array(train_losses)

plt.figure(figsize=(10, 6))

# Основная линия — функция потерь
plt.plot(epochs, loss_values, color='blue', linestyle='-', label='Loss')

# Точки (итерации AdaBoost)
plt.scatter(epochs, loss_values, color='red', zorder=5, label='AdaBoost Step')

# Стрелки — движение loss
for i in range(1, len(epochs)):
    plt.annotate('',
                 xy=(epochs[i], loss_values[i]),
                 xytext=(epochs[i - 1], loss_values[i - 1]),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Оформление графика
plt.title("Функция потерь и шаги AdaBoost")
plt.xlabel("Итерация (число базовых моделей)")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


epochs = np.arange(1, len(train_accs) + 1)
train_vals = np.array(train_accs)
test_vals = np.array(test_accs)

plt.figure(figsize=(10, 6))

# Линии точности
plt.plot(epochs, train_vals, color='green', linestyle='-', label='Train Accuracy')
plt.plot(epochs, test_vals, color='orange', linestyle='-', label='Test Accuracy')

# Точки
plt.scatter(epochs, train_vals, color='green', s=30, zorder=5)
plt.scatter(epochs, test_vals, color='orange', s=30, zorder=5)

# Стрелки на тестовой метрике
for i in range(1, len(epochs)):
    plt.annotate('',
                 xy=(epochs[i], test_vals[i]),
                 xytext=(epochs[i - 1], test_vals[i - 1]),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Оформление графика
plt.title("Динамика Accuracy по шагам AdaBoost")
plt.xlabel("Итерация (число базовых моделей)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
