import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# load data
from helper_code import twelve_leads

fre = 500

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

# import dataset
LAD_recordings = np.load('../data/LAD_data.npy')

# 1. Preprocessing
# down sample
LAD_reshaped = signal.resample(LAD_recordings, 200 * 10, axis=2)

idx = 0
# vis
for x in LAD_reshaped:
    if idx == 5:
        break
    b, a = butterBandPassFilter(0.05, 30, fre, order=4)
    x = signal.lfilter(b, a, x)

    # 消除线性趋势，解决基线漂移问题
    x = signal.detrend(x)

    # # 进行带阻滤波
    # b, a = butterBandStopFilter(48, 52, fre, order=2)
    # x = signal.lfilter(b, a, x)
    plt.figure(figsize=(20, 43))
    for i in range(0, 12):
        plt.subplot(12, 1, i + 1)
        # Show 10 second img
        # Needs to distinguish different dataset to get frequency
        # fre changed to 200
        plt.plot(np.arange(0, 200 * 10) / 200, x[i, : 200 * 10])
        plt.title(twelve_leads[i])
        plt.xticks(np.arange(0, 10, 0.4))

    plt.savefig(fname="./vis_LAD/down_sample_and_baseline" + str(idx) + ".png", dpi=300)
    plt.clf()
    idx += 1
    # plt.show()

LAD_reshaped = signal.detrend(LAD_reshaped)
np.save('../data/LAD_data_new.npy', np.array(LAD_reshaped))

# 2. QRS detection
