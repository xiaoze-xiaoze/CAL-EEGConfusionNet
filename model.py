import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx
from torch.utils.data import TensorDataset, DataLoader

def preprocess(data, channel_std=None):
    # 基线校正
    data = data - np.mean(data, axis=2, keepdims=True)

    # 计算或使用通道标准差
    if channel_std is None:
        _, n_channels, _ = data.shape
        data_reshaped = data.transpose(1, 0, 2).reshape(n_channels, -1)
        channel_std = np.std(data_reshaped, axis=1)
        channel_std = np.where(channel_std == 0, 1.0, channel_std)

    # 标准化
    data = data / channel_std.reshape(1, -1, 1)

    return data, channel_std


class GLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class ChannelAttention(nn.Module):
    def __init__(self, n_channels, reduction=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction),
            nn.ReLU(),
            nn.Linear(n_channels // reduction, n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, time)
        pooled = x.mean(dim=2)  # (batch, channels)
        weights = self.fc(pooled)  # (batch, channels)
        return x * weights.unsqueeze(2)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, time, hidden)
        weights = self.attention(x)  # (batch, time, 1)
        weights = F.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)  # (batch, hidden)


class GAT(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.head_dim = out_features // n_heads

        assert out_features % n_heads == 0

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape

        h = self.W(x)
        h = h.view(B, N, self.n_heads, self.head_dim)

        attn_src = (h * self.a_src).sum(dim=-1)
        attn_dst = (h * self.a_dst).sum(dim=-1)

        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)
        attn = self.leakyrelu(attn)
        attn = torch.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out_list = []
        for head in range(self.n_heads):
            attn_head = attn[:, :, :, head]
            h_head = h[:, :, head, :]
            out_head = torch.bmm(attn_head, h_head)
            out_list.append(out_head)

        out = torch.cat(out_list, dim=-1)

        return out


class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_dim, input_dim)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        gate = torch.sigmoid(self.gate(residual))
        x = gate * x + (1 - gate) * residual
        return self.ln(x)


class EEGConfusionNet(nn.Module):
    def __init__(self, n_channels=8, seq_len=250, n_classes=3, conv_filters=64, lstm_hidden=128, dropout=0.3, gat_heads=4):
        super().__init__()

        self.conv3 = nn.Conv1d(n_channels, conv_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(n_channels, conv_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(n_channels, conv_filters, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(conv_filters * 3)

        self.glu = GLU(conv_filters * 3)
        self.dropout1 = nn.Dropout(dropout)

        self.spatial_gat = GAT(conv_filters * 3, conv_filters * 3, n_heads=gat_heads, dropout=dropout)

        self.channel_attn = ChannelAttention(conv_filters * 3)

        self.lstm = nn.LSTM(conv_filters * 3, lstm_hidden,
                           batch_first=True, bidirectional=True)

        self.temporal_attn = TemporalAttention(lstm_hidden * 2)

        self.grn = GRN(lstm_hidden * 2, lstm_hidden, dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden, n_classes)
        )

    def forward(self, x):
        B, C, T = x.shape

        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        c7 = F.relu(self.conv7(x))
        x = torch.cat([c3, c5, c7], dim=1)
        x = self.bn(x)

        x = x.transpose(1, 2)
        x = self.glu(x)
        x = self.dropout1(x)

        x = self.spatial_gat(x)

        x = x.transpose(1, 2)

        x = self.channel_attn(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm(x)

        x = self.temporal_attn(x)

        x = self.grn(x)

        x = self.classifier(x)

        return x

def train(data_x, data_y, epochs=30, lr=0.001, batch_size=64, val_split=0.2, device='cuda'):
    data_x = torch.FloatTensor(data_x)
    data_y = torch.LongTensor(data_y).squeeze()

    dataset = TensorDataset(data_x, data_y)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EEGConfusionNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}: \nTrain Loss: {train_loss:.4f}  Acc: {train_acc:.4f} \nVal Loss: {val_loss:.4f} Acc: {val_acc:.4f} \n')

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(history['train_loss'], label='Train Loss', c = 'blue')
    ax1.plot(history['val_loss'], label='Val Loss', c = 'red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history['train_acc'], label='Train Acc', c = 'green')
    ax2.plot(history['val_acc'], label='Val Acc', c = 'orange')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return model, history


def final(data_x, data_y, epochs=30, lr=0.001, batch_size=64, device='cuda'):
    data_x = torch.FloatTensor(data_x)
    data_y = torch.LongTensor(data_y).squeeze()

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(data_x, data_y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EEGConfusionNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}: \nTrain Loss: {train_loss:.4f}  Acc: {train_acc:.4f} \n')

    return model


def save_model(model, device, save_path=r'D:\Deep Learning\CAL-EEG database for Confusion Analysis in Learning\Code\model\EEGConfusionNet.onnx'):
    model.eval()
    dummy_input = torch.randn(1, 8, 250).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("模型已保存")

if __name__ == '__main__':
    data_x = np.load(r'D:\Deep Learning\CAL-EEG database for Confusion Analysis in Learning\Data\2025CALx_train.npy', allow_pickle=True)
    data_y = np.load(r'D:\Deep Learning\CAL-EEG database for Confusion Analysis in Learning\Data\2025CALy_train.npy', allow_pickle=True)
    data_x, channel_std = preprocess(data_x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, history = train(data_x, data_y, epochs=30, lr=0.001, batch_size=64, val_split=0.2, device=device)

    final_model = final(data_x, data_y, epochs=30, lr=0.001, batch_size=64, device=device)

    save_model(final_model, device, r'D:\Deep Learning\CAL-EEG database for Confusion Analysis in Learning\Code\model\EEGConfusionNet_final.onnx')