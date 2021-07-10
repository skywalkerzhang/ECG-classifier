import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy import signal

from helper_code import find_challenge_files, load_header, load_recording, get_labels, twelve_leads

PR_thresh = 0.18
mis_data = []
mis_labels = []
fre = 500
idx = 0

def butterBandPassFilter(lowcut, highcut, samplerate, order):
    "生成巴特沃斯带通滤波器"
    semiSampleRate = samplerate * 0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b, a = signal.butter(order, [low, high], btype='bandpass')
    # print("bandpass:", "b.shape:", b.shape, "a.shape:", a.shape, "order=", order)
    # print("b=", b)
    # print("a=", a)
    return b, a


def butterBandStopFilter(lowcut, highcut, samplerate, order):
    "生成巴特沃斯带阻滤波器"
    semiSampleRate = samplerate * 0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b, a = signal.butter(order, [low, high], btype='bandstop')
    # print("bandstop:", "b.shape:", b.shape, "a.shape:", a.shape, "order=", order)
    # print("b=", b)
    # print("a=", a)
    return b, a

truth = pd.read_csv("../data/test_split0.csv")
PR_truth = truth['10370003']
pred = pd.read_csv("../data/test_split0_pred.csv")
PR_pred = pred['10370003']

thresh = np.load("../data/magic_weight_avg.npz")
print(thresh['arr_0'][0])

PR_truth = np.array(PR_truth, dtype=np.float32)
PR_pred = np.array(PR_pred, dtype=np.float32)

for i in range(0, len(PR_pred)):
    if PR_pred[i] < PR_thresh:
        PR_pred[i] = 0
    else:
        PR_pred[i] = 1

mis_idx = []
for i in range(0, len(PR_pred)):
    if PR_truth[i] != PR_pred[i] and PR_pred[i] == 1:
        mis_idx.append(i)

names = truth.iloc[mis_idx, [0]]['filename']
print(names)
names = names.apply(lambda x: x[8: -4])
names = np.array(names)

C = confusion_matrix(np.array(PR_truth), np.array(PR_pred))

print(C)

# vis
data_directory_1 = "../data/WFDB_CPSC2018_2"
data_directory_2 = "../data/WFDB_PTBXL"

header = load_header("../data/WFDB_CPSC2018\\Q3458.hea")
recording = load_recording("../data/WFDB_CPSC2018_2\\Q3458.mat")
current_labels = get_labels(header)
mis_data.append(recording)
mis_labels.append(current_labels)

header = load_header("../data/WFDB_PTBXL\\HR00377.hea")
recording = load_recording("../data/WFDB_PTBXL\\HR00377.mat")
current_labels = get_labels(header)
mis_data.append(recording)
mis_labels.append(current_labels)
header = load_header("../data/WFDB_PTBXL\\HR04377.hea")
recording = load_recording("../data/WFDB_PTBXL\\HR04377.mat")
current_labels = get_labels(header)
mis_data.append(recording)
mis_labels.append(current_labels)

# 进行带通滤波

for x in mis_data:
    # b, a = butterBandPassFilter(3, 70, fre, order=4)
    # x = signal.lfilter(b, a, x)
    #
    # # 进行带阻滤波
    # b, a = butterBandStopFilter(48, 52, fre, order=2)
    # x = signal.lfilter(b, a, x)
    plt.figure(figsize=(20, 43))
    for i in range(0, 12):
        plt.subplot(12, 1, i + 1)
        # Show 10 second img
        # Needs to distinguish different dataset to get frequency
        plt.plot(np.arange(0, fre * 10) / fre, x[i, : fre * 10])
        plt.title(twelve_leads[i])
        plt.xticks(np.arange(0, 10, 0.4))

    plt.savefig(fname="./vis_PR/mis/" + names[idx] + ".png", dpi=300)
    plt.clf()
    idx += 1
    # plt.show()

