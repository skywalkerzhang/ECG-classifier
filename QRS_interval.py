import neurokit2 as nk
import numpy as np
from wfdb import processing
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

from helper_code import load_recording

LAD_r_peaks = []
LAD_waves_dwt = []
LAD_interval = []


# import data: train split & test split

# feature selection
# get all sum of QRS
leads = {0: 'I', 1: 'II', 2: 'III', 3: 'aVR', 4: 'aVL', 5: 'aVF', 6: 'V1', 7: 'V2', 8: 'V3', 9: 'V4', 10: 'V5', 11: 'V6'}
# get the average length of interval

# Retrieve ECG data from data folder (sampling rate= 1000 Hz)
LAD_recordings = np.load('../data/training_sets0.npy')
for x in LAD_recordings:
    print('I am running')
    each_peaks = []
    each_waves = []
    each_inter = []
    for i in leads.keys():
        print(i)
        ecg_signal = x[i]
        # Extract R-peaks locations
        # clean ecg
        ecg_signal = nk.ecg_clean(ecg_signal, 257)
        r_peaks = None
        try:
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=257)
            if len(r_peaks['ECG_R_Peaks']) == 0:
                each_inter.append(0)
                each_inter.append(0)
                nk.signal_plot(ecg_signal, 257)
                print(idx, 'wrong')
                plt.show()
                continue
        except:
            nk.signal_plot(ecg_signal, 257)
            print(idx)
            plt.show()
        # 返回数据下标
        # r_peaks = processing.xqrs_detect(ecg_signal, fs)
        each_peaks.append(r_peaks)
        # # Visualize R-peaks in ECG signal
        # plt.figure()
        # plt.plot(ecg_signal)
        # # plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'ro')
        # plt.plot(rpeaks['ECG_R_Peaks'], ecg_signal[rpeaks['ECG_R_Peaks']], 'ro')
        # plt.title('Detected R-peaks on lead' + str(i))
        #
        # plt.show()

        # # Delineate the ECG signal
        # Delineate the ECG signal and visualizing all peaks of ECG complexes
        signal_dwt, waves_dwt = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=257, method="dwt", show_type='bounds_R')
        R_Onsets = waves_dwt['ECG_R_Onsets']
        R_Offsets = waves_dwt['ECG_R_Offsets']
        s = 0
        # the sum of QRS
        for j in range(len(R_Onsets)):
            if math.isnan(R_Onsets[j]):
                R_Onsets[j] = 0
            if math.isnan(R_Offsets[j]):
                R_Offsets[j] = len(R_Onsets) - 1
            for k in range(int(R_Onsets[j]), int(R_Offsets[j]) + 1):
                s += ecg_signal[k]

        each_inter.append(s)
        each_inter.append(sum(R_Offsets) - sum(R_Onsets) / len(R_Onsets))

        # the average length of QRS

        
    LAD_r_peaks.append(each_peaks)
    LAD_interval.append(each_inter)

np.save('../data/LAD_train.npy', np.array(LAD_interval))