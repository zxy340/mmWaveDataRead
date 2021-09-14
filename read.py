import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D

B = 3753.75e6  # Sweep Bandwidth
T = 250e-6  # Sweep Time
N = 256  # Sample Length
L = 190  # Chirp Total
c = 3e8  # Speed of Light
f0 = 77e9  # Start Frequency
NumRangeFFT = 256  # Range FFT Length
NumDopplerFFT = 190  # Doppler FFT Length
rangeRes = c/2/B  # Range Resolution
velRes = c/2/f0/T/NumDopplerFFT  # Velocity Resolution
maxRange = rangeRes * NumRangeFFT  # Max Range
maxVel = velRes * NumDopplerFFT/2  # Max Velocity
tarR = [50, 90]  # Target Range
tarV = [3, 20]  # Target Velocity

class AWR1642:
    def __init__(
        self,
        dfile,
        sample_rate,
        num_frame=320,
        num_chirp=190,
    ):
        num_channel = 4
        x = np.fromfile(dfile, dtype=np.int16)
        x = x.reshape(num_frame, num_chirp, num_channel, -1, 4)  # 2I + 2Q = 4
        # 关于IQ信号，可以参考https://sunjunee.github.io/2017/11/25/what-is-IQ-signal/
        x_I = x[:, :, :, :, :2].reshape(
            num_frame, num_chirp, num_channel, -1
        )  # flatten the last two dims of I data
        x_Q = x[:, :, :, :, 2:].reshape(
            num_frame, num_chirp, num_channel, -1
        )  # flatten the last two dims of Q data
        data = np.array((x_I, x_Q))  # data[I/Q, Frame, Chirp, Channel, Sample]
        self.data = np.transpose(
            data, (0, 3, 1, 2, 4)
        )  # data[I/Q, Channel, Frame, Chirp, Sample]
        self.sample_rate = sample_rate

dfile = "C:/Users/86182/OneDrive/桌面/1631301582.8862.bin"
mmWave_Data = AWR1642(dfile, 5000)
print(mmWave_Data.data.shape)

sigReceive = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    for n in range(0, N):
        sigReceive[l][n] = np.complex(0, 1) * mmWave_Data.data[0, 0, 0, l, n] + mmWave_Data.data[1, 0, 0, l, n]

# range win processing
sigRangeWin = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    sigRangeWin[l] = np.multiply(sigReceive[l], np.hamming(N).T)

# range fft processing
sigRangeFFT = np.zeros((L, N), dtype=complex)
for l in range(0, L):
    sigRangeFFT[l] = np.fft.fft(sigRangeWin[l], NumRangeFFT)

# doppler win processing
sigDopplerWin = np.zeros((L, N), dtype=complex)
for n in range(0, N):
    sigDopplerWin[:, n] = np.multiply(sigRangeFFT[:, n], np.hamming(L).T)

# doppler fft processing
sigDopplerFFT = np.zeros((L, N), dtype=complex)
for n in range(0, N):
    sigDopplerFFT[:, n] = np.fft.fftshift(np.fft.fft(sigDopplerWin[:, n], NumDopplerFFT))

fig = plt.figure()
# ax = Axes3D(fig)
#
x = np.arange(0, NumRangeFFT * rangeRes, rangeRes)
y = np.arange((-NumDopplerFFT / 2) * velRes, (NumDopplerFFT / 2) * velRes, velRes)
# # x = np.arange(NumRangeFFT)
# # y = np.arange(NumDopplerFFT)
# # print(len(x))
# # print(len(y))
X, Y = np.meshgrid(x, y)
Z = np.abs(sigDopplerFFT)
# ax.plot_surface(X, Y, Z,
#                 rstride=1,  # rstride（row）指定行的跨度
#                 cstride=1,  # cstride(column)指定列的跨度
#                 cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
#
# ax.invert_xaxis()  # x轴反向
#
plt.pcolor(X, Y, Z)
plt.show()