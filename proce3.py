import numpy as np
import matplotlib.pyplot as plt

N = 64
m = N / 8
K = 2 * m + 1
n = np.arange(-N / 2, N / 2)
h = (1 / N) * np.sin(np.pi * n * K / N) / (np.sin(np.pi * n / N) + 1e-20)
plt.stem(h)
plt.show()
h[int(N/2)] = K/N
w = np.hanning(N)
hw = h * w
H = np.fft.fft(hw, 2*N)

plt.stem(n, hw)
plt.show()

plt.stem(np.fft.fftshift(abs(H)))
plt.show()



n1 = np.arange(200)
x1 = np.sin(2*np.pi*(100/16000)*n1)
x2 = 0.2*np.sin(2*np.pi*(4000/16000)*n1)
xt = x1+x2
y = np.convolve(hw,xt)
plt.stem(y)
plt.show()