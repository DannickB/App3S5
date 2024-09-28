import numpy as np
import matplotlib.pyplot as plt

N = 50
n = np.arange(N)
m = np.arange(-N/2, N/2)
x1 = np.sin(0.1 * np.pi * n + (np.pi/4))

x2 = np.tile([1, -1], 10)

x3 = np.zeros(N)
x3[10] = 1

X1 = np.fft.fft(x1)
plt.stem(m, np.fft.fftshift(abs(X1)))

plt.show()
#plt.stem(np.angle(X1))
plt.show()
