import neurokit2 as nk
import numpy as np
import math


# import data: train split & test split

# feature selection
# get all sum of QRS
leads = {0: 'I', 1: 'II', 2: 'III', 3: 'aVR',
        4: 'aVL', 5: 'aVF', 6: 'V1', 7: 'V2',
        8: 'V3', 9: 'V4', 10: 'V5', 11: 'V6'}
# get the average length of interval

# Retrieve ECG data from data folder (sampling rate= 1000 Hz)
LAD_recordings = np.load('../data/test_sets0.npy')
print('process data, data size: ', len(LAD_recordings))
LAD_features = []
for recording in LAD_recordings:
    print('running', len(LAD_features))
    # store features of 12 leads
    leads_features = []
    for lead_idx in leads.keys():
        # store three features
        lead_features = []
        ecg_signal = recording[lead_idx]

        # Extract R-peaks locations
        # clean ecg
        ecg_signal = nk.ecg_clean(ecg_signal, 257)
        # get r peaks of a signal
        r_peaks = None
        try:
            _, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=257)
            if len(r_peaks['ECG_R_Peaks']) == 0:
                lead_features.append(0)
                lead_features.append(0)
                leads_features.append(lead_features)
                continue
        except:
            lead_features.append(0)
            lead_features.append(0)
            leads_features.append(lead_features)
            continue

        # 返回数据下标
        # r_peaks = processing.xqrs_detect(ecg_signal, fs)
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

        # the sum of QRS
        s = 0
        for j in range(len(R_Onsets)):
            if math.isnan(R_Onsets[j]):
                R_Onsets[j] = 0
            if math.isnan(R_Offsets[j]):
                R_Offsets[j] = len(R_Onsets) - 1
            for k in range(int(R_Onsets[j]), int(R_Offsets[j]) + 1):
                s += ecg_signal[k]

        lead_features.append(s)
        lead_features.append(sum(R_Offsets) - sum(R_Onsets) / len(R_Onsets))
        
        leads_features.append(lead_features)

    LAD_features.append(leads_features)
        

np.save('../data/LAD_train.npy', np.array(LAD_features))
