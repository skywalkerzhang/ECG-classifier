import numpy as np
import matplotlib.pyplot as plt
from wfdb import processing
from ecg_detectors import Detectors

leads = {0: 'I', 2: 'III', 5: 'aVF'}
LAD_recordings = np.load('../data/LAD_data_new.npy')
tmp = LAD_recordings[0]
for i in leads.keys():
    unfiltered_ecg = tmp[i]
    fs = 200
    # 返回数据下标
    r_peaks = processing.xqrs_detect(unfiltered_ecg, fs)
    # all_peaks = processing.find_peaks(unfiltered_ecg)
    detectors = Detectors(fs)

    # r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg)

    plt.figure()
    plt.plot(unfiltered_ecg)
    plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'ro')
    # plt.plot(all_peaks[0], unfiltered_ecg[all_peaks[0]], 'ro')
    plt.title('Detected R-peaks on lead' + leads[i])

    plt.show()

'''
for x in LAD_recordings:
    unfiltered_ecg = x[0]
    fs = 200

    detectors = Detectors(fs)
    
    #r_peaks = detectors.two_average_detector(unfiltered_ecg)
    #r_peaks = detectors.matched_filter_detector(unfiltered_ecg,"templates/template_250hz.csv")
    #r_peaks = detectors.swt_detector(unfiltered_ecg)
    r_peaks = detectors.engzee_detector(unfiltered_ecg)
    #r_peaks = detectors.christov_detector(unfiltered_ecg)
    #r_peaks = detectors.hamilton_detector(unfiltered_ecg)
    #r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg)


    plt.figure()
    plt.plot(unfiltered_ecg)
    plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'ro')
    plt.title('Detected R-peaks')
    
    plt.show()
'''
