'''
This code is used for generating LAD dataset and visualization
'''
from helper_code import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

dataset_names = ['WFDB_CPSC2018', 'WFDB_CPSC2018_2', 'WFDB_Ga', 'WFDB_PTB', 'WFDB_PTBXL', 'WFDB_StPetersburg']
fre_500_dataset = ['WFDB_CPSC2018', 'WFDB_CPSC2018_2', 'WFDB_Ga', 'WFDB_PTBXL']
fre = 500
times = []

labels = []
LAD_dataset = []
LAD_recordings = []
all_recordings = []
idx_fre = 0
idx_LAD = 0

# generate PR dataset
'''
for dataset_name in dataset_names:
    data_directory = "../data/" + dataset_name
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    for i in range(num_recordings):
        # print('    {}/{}...'.format(i + 1, num_recordings))

        # Load header and recording.
        # Header contains the label, sex, age and sur...
        header = load_header(header_files[i])
        # Have leads information.
        recording = load_recording(recording_files[i])

        # get times for each recordings
        time = float(get_num_samples(header)) / float(get_frequency(header))

        # get labels that contains PR
        # generate the visual graph
        current_labels = get_labels(header)
        if '39732003' in current_labels:
            idx_LAD += 1
            LAD_dataset.append([header, recording, time])
            if len(recording[0]) < 5000:
                continue
            LAD_recordings.append(recording[:, :5000])
'''


# preprocess dataset to make it uniform
# only pick 10 seconds data
# Because they have same frequency so just truncate the recordings

# np.save('../data/LAD_data.npy', np.array(LAD_recordings))


# static the number of PR classes
print(len(LAD_dataset))
# generate PR dataset

for dataset_name in fre_500_dataset:
    data_directory = "../data/" + dataset_name
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    for i in range(num_recordings):
        # print('    {}/{}...'.format(i + 1, num_recordings))

        # Load header and recording.
        # Header contains the label, sex, age and sur...
        header = load_header(header_files[i])
        # Have leads information.
        recording = load_recording(recording_files[i])

        # get times for each recordings
        time = float(get_num_samples(header)) / float(get_frequency(header))

        # get labels that contains LAD
        # generate the visual graph
        current_labels = get_labels(header)
        if not np.any(recording):
            continue
        if len(recording[0]) < 5000:
            continue
        if '39732003' in current_labels:
            labels.append(1)
        else:
            labels.append(0)

        all_recordings.append(recording[:, :5000])
print(len(labels))
print(len(all_recordings))
np.save('../data/all_data_500.npy', np.array(all_recordings))
np.save('../data/labels_500.npy', np.array(labels))
'''
# Visualization
for x in LAD_dataset:
    recording_vis = x[1]
    # use header's name as file name.
    header_name = x[0].split(' ')[0]
    plt.figure(figsize=(20, 43))
    for i in range(0, 12):
        plt.subplot(12, 1, i + 1)
        # Show 10 second img
        # Needs to distinguish different dataset to get frequency
        if header_name[0] == 'S':
            fre = 1000
        elif header_name[0] == 'I':
            fre = 257

        plt.plot(np.arange(0, fre * 10) / fre, recording_vis[i, : fre * 10])
        plt.title(twelve_leads[i])
        plt.xticks(np.arange(0, 10, 0.4))

    plt.savefig(fname="./vis_LAD/" + header_name[:-4] + ".png", dpi=300)
    plt.clf()
        # plt.show()
'''

