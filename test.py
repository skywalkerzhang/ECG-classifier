import neurokit2 as nk
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as scisig

LAD_recordings = np.load('../data/test_sets0.npy')

for recording in LAD_recordings:
    # iterate 12 leads
    b, a = scisig.butter(4, 0.0003, btype='highpass')  
    for lead in recording:
        filtered = scisig.filtfilt(b, a, lead)  
        lead, _ = nk.ecg_process(lead, 257)
        nk.ecg_plot(lead)
        plt.show()
        break

    break

