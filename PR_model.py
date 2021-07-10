import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# All data are 500Hz
fre = 500
# import dataset
PR_recordings = np.load('../data/LAD_data.npy')

# preprocessing signals
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

plt.figure(figsize=(12, 5))
ax0 = plt.subplot(121)
for k in [2, 3, 4]:
    b, a = butterBandPassFilter(3, 70, samplerate=fre, order=k)
    w, h = signal.freqz(b, a, worN=2000)
    ax0.plot((fre * 0.5 / np.pi) * w, np.abs(h), label="order = %d" % k)

ax1 = plt.subplot(122)
for k in [2, 3, 4]:
    b, a = butterBandStopFilter(48, 52, samplerate=fre, order=k)
    w, h = signal.freqz(b, a, worN=2000)
    ax1.plot((fre * 0.5 / np.pi) * w, np.abs(h), label="order = %d" % k)

for x in PR_recordings:
    # Convert to frequency dominated space
    # fre = 500
    # np.fft.rfft(PR_recordings) 表示0hz 到 采样频率 / 2的频率范围对应的能量
    x_FFT = np.abs(np.fft.rfft(x) / fre)
    x_freqs = np.linspace(0, fre/2, int(fre / 2) * 10 + 1)

    # 消除线性趋势，解决基线漂移问题
    x = signal.detrend(x)

    # 进行带通滤波
    b, a = butterBandPassFilter(3, 70, fre, order=4)
    x = signal.lfilter(b, a, x)

    #
    # # 进行带阻滤波
    # b, a = butterBandStopFilter(48, 52, fre, order=2)
    # x = signal.lfilter(b, a, x)


    # vis
    plt.figure(figsize=(10, 6))
    ax0 = plt.subplot(211)             #画时域信号
    ax0.set_xlabel("Time(s)")
    ax0.set_ylabel("Amp(μV)")
    # only choose lead2
    ax0.plot(np.arange(0, 10, 10/5000), x[1])
    # ax0.set_xticklabels(np.arange(0, 10, 0.4))

    ax1 = plt.subplot(212)             #画频域信号-频谱
    ax1.set_xlabel("Freq(Hz)")
    ax1.set_ylabel("Power")
    ax1.plot(x_freqs, x_FFT[1])
    plt.show()


