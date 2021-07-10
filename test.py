import neurokit2 as nk
import numpy as np
from matplotlib import pyplot as plt
ecg_signal = np.loadtxt('./test.txt')
_, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=257)
print(r_peaks)
