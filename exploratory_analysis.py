import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_x = np.load(r'D:\Deep Learning\CAL-EEG database for Confusion Analysis in Learning\Data\2025CALx_train.npy', allow_pickle=True)
data_y = np.load(r'D:\Deep Learning\CAL-EEG database for Confusion Analysis in Learning\Data\2025CALy_train.npy', allow_pickle=True)

def descriptive(data_x, data_y):
    print(data_x.shape)
    print(data_x.dtype)
    print(data_y.shape)
    print(data_y.dtype)
    confused_count = np.sum(data_y == 0)
    not_confused_count = np.sum(data_y == 1)
    other_count = np.sum(data_y == 2)
    print(f"Confused: {confused_count} \n Not Confused: {not_confused_count} \n Other: {other_count}")
    
def plot(data_x, data_y, sample_idx):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    sample_data = data_x[sample_idx]
    sample_label = int(data_y[sample_idx].item())

    emotion_map = {
        0: "Confused",
        1: "Not Confused",
        2: "Other"
    }
    emotion_name = emotion_map.get(sample_label, f"Unknown({sample_label})")

    channel_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']

    emotion_colors = {0: 'red', 1: 'blue', 2: 'green'}
    emotion_color = emotion_colors.get(sample_label, 'gray')

    time_points = np.arange(sample_data.shape[1])

    for channel in range(8):
        ax.plot(time_points,
                [channel] * len(time_points),
                sample_data[channel, :],
                color=channel_colors[channel],
                alpha=0.8,
                linewidth=2,
                label=f'Ch{channel+1}')
        
    z_min = sample_data.min()
    z_max = sample_data.max()
    z_range = z_max - z_min

    emotion_positions = {
        0: z_min + 0.25 * z_range,
        1: z_min + 0.50 * z_range,
        2: z_min + 0.75 * z_range
    }
    emotion_value = emotion_positions[sample_label]
    emotion_line = np.full(len(time_points), emotion_value)

    ax.plot(time_points,
            [8] * len(time_points),
            emotion_line,
            color=emotion_color,
            alpha=0.9,
            linewidth=1.5,
            label=f'Emotion: {emotion_name} (Label={sample_label})',
            linestyle='-')

    ax.set_xlabel('Time Points', fontsize=12, labelpad=10)
    ax.set_ylabel('Channel', fontsize=12, labelpad=10)
    ax.set_zlabel('Amplitude / Emotion Label', fontsize=12, labelpad=10)

    ax.set_yticks(range(9))
    ax.set_yticklabels([f'Ch{i+1}' for i in range(8)] + ['Emotion'])

    ax.legend(loc='upper left', fontsize=9, ncol=3)

    ax.view_init(elev=20, azim=45)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()

if __name__ == '__main__':
    # descriptive(data_x, data_y)
    plot(data_x, data_y, sample_idx=2800)
