import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal

data_x = np.load(r'D:\Deep Learning Project\CAL-EEGConfusionNet\Data\2025CALx_train.npy', allow_pickle=True)
data_y = np.load(r'D:\Deep Learning Project\CAL-EEGConfusionNet\Data\2025CALy_train.npy', allow_pickle=True)

def descriptive(data_x, data_y):
    print(data_x.shape)
    print(data_x.dtype)
    print(data_y.shape)
    print(data_y.dtype)
    confused_count = np.sum(data_y == 0)
    not_confused_count = np.sum(data_y == 1)
    other_count = np.sum(data_y == 2)
    print(f"Confused: {confused_count} \nNot Confused: {not_confused_count} \nOther: {other_count}")
    
def plot(data_x, sample_idx):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    sample_data = data_x[sample_idx]

    channel_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']

    time_points = np.arange(sample_data.shape[1])

    for channel in range(8):
        ax.plot(time_points,
                [channel] * len(time_points),
                sample_data[channel, :],
                color=channel_colors[channel],
                alpha=0.8,
                linewidth=2,
                label=f'Ch{channel+1}')

    ax.set_xlabel('Time Points', fontsize=12, labelpad=10)
    ax.set_ylabel('Channel', fontsize=12, labelpad=10)
    ax.set_zlabel('Amplitude', fontsize=12, labelpad=10)

    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Ch{i+1}' for i in range(8)])
    ax.set_ylim(-0.5, 7.5)

    ax.legend(loc='upper left', fontsize=9, ncol=3)

    ax.view_init(elev=20, azim=45)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()

def bandpass_filter(data, lowcut=0.5, highcut=50, fs=250, order=4):
    # Butterworth带通滤波，保留0.5-50Hz频段
    nyquist = fs / 2
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered_data[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
    return filtered_data

def notch_filter(data, freq=50, fs=250, quality=30):
    # 陷波滤波，去除50Hz工频噪声
    b, a = signal.iirnotch(freq, quality, fs)
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered_data[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
    return filtered_data

def preprocess(data, fs=250):
    data = bandpass_filter(data, lowcut=0.5, highcut=50, fs=fs)
    data = notch_filter(data, freq=50, fs=fs)
    data = data - np.mean(data, axis=2, keepdims=True)
    _, n_channels, _ = data.shape
    data_reshaped = data.transpose(1, 0, 2).reshape(n_channels, -1)
    channel_std = np.std(data_reshaped, axis=1)
    channel_std = np.where(channel_std == 0, 1.0, channel_std)
    data = data / channel_std.reshape(1, -1, 1)
    return data, channel_std

if __name__ == '__main__':
    # descriptive(data_x, data_y)
    # plot(data_x, sample_idx=2800)

    processed_data, channel_std = preprocess(data_x, fs=250)
    plot(processed_data, sample_idx=2800)

    np.save(r'D:\Deep Learning Project\CAL-EEGConfusionNet\Data\2025CALx_train_processed.npy', processed_data)
    np.save(r'D:\Deep Learning Project\CAL-EEGConfusionNet\Data\channel_std.npy', channel_std)