from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper_code import twelve_leads, load_header, load_recording, get_labels

fre = 500
idx = 0
s = 0

def get_confusion_value(y_true, y_pred):
    # ordering: tn, fp, fn, tp
    return confusion_matrix(y_true, y_pred).ravel()

def file_to_numpy(file_path):
    ext = file_path.split('.')[-1]
    if ext == 'csv':
        return np.genfromtxt(file_path, delimiter=',')
    elif ext == 'npz' or ext == 'npy':
        return np.load(file_path)
    else:
        raise Exception('Unsupported file type')

def get_truth_by_thresh(labels, thresh):
    return np.array([l for l in np.greater_equal(labels, thresh)]).astype(int)

files = [
    x for x in map(lambda v: f'../data/predict_label{v}.npz', range(5))
]

thresh = file_to_numpy('../data/magic_weight_avg.npz')['arr_0'][0]

all_tp = 0
all_fp = 0
all_fn = 0
all_tn = 0
for file in files:
    labels = file_to_numpy(file)
    label_true = labels['arr_0']
    label_pred = get_truth_by_thresh(labels['arr_1'], thresh)

    # get the label of LAD
    lad_true = label_true[:, 11]
    lad_pred = label_pred[:, 11]

    # calculate confusion matrix
    tn, fp, fn, tp = get_confusion_value(lad_true, lad_pred)
    all_tp += tp
    all_tn += tn
    all_fp += fp
    all_fn += fn
    print(f'tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}')

s = all_fp + all_fn + all_tn + all_tp
print("s:", s)
acc = (all_tn + all_tp) / (all_fp + all_fn + all_tn + all_tp)
print("all_fp:", all_fp, "all_fn:", all_fn, "all_tn:", all_tn, "all_tp:", all_tp)
print(acc)

# code for misclassification
FP_idx = []
FN_idx = []
for i in range(0, len(lad_pred)):
    if lad_true[i] != lad_pred[i] and lad_pred[i] == 1:
        FP_idx.append(i)
    elif lad_true[i] != lad_pred[i] and lad_pred[i] == 0:
        FN_idx.append(i)

names = pd.read_csv("../data/test_split4.csv")
FP_names = names.iloc[FP_idx, [0]]['filename']
FP_names = FP_names.apply(lambda x: x[8: -4])
FP_names = np.array(FP_names)

FN_names = names.iloc[FN_idx, [0]]['filename']
FN_names = FN_names.apply(lambda x: x[8: -4])
FN_names = np.array(FN_names)

FPs = []
FNs = []

for i in range(20):
    header = load_header("../data/WFDB_PTBXL\\" + FN_names[i] + ".hea")
    recording = load_recording("../data/WFDB_PTBXL\\" + FN_names[i] + ".mat")
    current_labels = get_labels(header)
    FNs.append(recording)

    header = load_header("../data/WFDB_CPSC2018\\" + FP_names[i] + ".hea")
    recording = load_recording("../data/WFDB_CPSC2018\\" + FP_names[i] + ".mat")
    current_labels = get_labels(header)
    FPs.append(recording)

for x in FPs:
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

    plt.savefig(fname="./vis_LAD/FPs/" + FP_names[idx] + ".png", dpi=300)
    plt.clf()
    idx += 1
    # plt.show()

idx = 0
for x in FNs:
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

    plt.savefig(fname="./vis_LAD/FNs/" + FN_names[idx] + ".png", dpi=300)
    plt.clf()
    idx += 1
    # plt.show()