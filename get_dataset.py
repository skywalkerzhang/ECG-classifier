import pandas as pd
from helper_code import load_recording
import numpy as np
import neurokit2 as nk
from matplotlib import pyplot as plt
all_training_recordings = []
training_labels = []


test_labels = []
all_test_recordings = []

train_files = [
    x for x in map(lambda v: f'../data/train_split{v}.csv', range(5))
]

test_files = [
    x for x in map(lambda v: f'../data/test_split{v}.csv', range(5))
]

# index corresponding to the number of training set
for i in range(len(train_files)):
    train_file = train_files[i]
    training_recordings = []
    df = pd.read_csv(train_file)
    file_name = df['filename']
    file_name = file_name.apply(lambda x: x[8:])
    for f in file_name:
        mark = f[0]
        if mark == 'A':
            dataset = 'WFDB_CPSC2018'
            fre = 500
        elif mark == 'Q':
            dataset = 'WFDB_CPSC2018_2'
            fre = 500
        elif mark == 'E':
            dataset = 'WFDB_Ga'
            fre = 500
        elif mark == 'S':
            dataset = 'WFDB_PTB'
            fre = 1000
        elif mark == 'H':
            dataset = 'WFDB_PTBXL'
            fre = 500
        elif mark == 'I':
            dataset = 'WFDB_StPetersburg'
            fre = 257
        recording = load_recording('../data/' + dataset + '/' + f)
        # down sample
        recording_new = []
        for k in range(12):
            resampled = nk.signal_resample(recording[k], 2570, fre, 257)
            recording_new.append(resampled)

        training_recordings.append(np.array(recording_new))
    np.save(f'../data/training_sets{i}.npy', np.array(training_recordings))
    current_label = df['39732003']
    np.save(f'../data/training_labels{i}.npy', current_label.to_numpy())

# test
for i in range(len(test_files)):
    test_file = test_files[i]
    test_recordings = []
    df = pd.read_csv(test_file)
    file_name = df['filename']
    file_name = file_name.apply(lambda x: x[8:])
    for f in file_name:
        mark = f[0]
        if mark == 'A':
            dataset = 'WFDB_CPSC2018'
            fre = 500
        elif mark == 'Q':
            dataset = 'WFDB_CPSC2018_2'
            fre = 500
        elif mark == 'E':
            dataset = 'WFDB_Ga'
            fre = 500
        elif mark == 'S':
            dataset = 'WFDB_PTB'
            fre = 1000
        elif mark == 'H':
            dataset = 'WFDB_PTBXL'
            fre = 500
        elif mark == 'I':
            dataset = 'WFDB_StPetersburg'
            fre = 257
        recording = load_recording('../data/' + dataset + '/' + f)
        # down sample
        recording_new = []
        for k in range(12):
            recording_new.append(nk.signal_resample(recording[k], 2570, fre, 257))
        test_recordings.append(recording_new)
    # np.save(f'../data/test_sets{i}.npy', np.array(test_recordings))
    # current_label = df['39732003']
    # np.save(f'../data/test_labels{i}.npy', current_label.to_numpy())
