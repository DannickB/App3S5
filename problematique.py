import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def extract_params(H):
    amp = []
    ph = []
    for i in range(1, 33):
        amp.append(abs(H[466 * i]))
        ph.append(np.angle(H[466 * i]))
    return amp, ph


def find_order(freq, target):
    N = 0
    Xn = 1
    T = 0
    while Xn - target > 0:
        N += 1
        T += np.exp(-1j * freq * N)
        Xn = np.abs(T / N)
    return N


def synthesis_note(N, amps, ph, freq, env):
    n = np.arange(N)
    x = np.zeros(N)
    for i in range(32):
        x += amps[i] * np.sin(freq * (i + 1) * n + ph[i])
    return x * env


if __name__ == "__main__":
    x, fe = sf.read('note_guitare_lad.wav')
    f = 14912
    N = len(x)
    w = np.hanning(N)
    xw = x * w
    X = np.fft.fft(xw, N)
    amp, ph = extract_params(X)

    filter_freq = np.pi / 1000
    target_gain = np.sqrt(2) / 2
    Nf = find_order(filter_freq, target_gain)
    Nf = 886
    nf = np.arange(-Nf / 2, Nf / 2)
    k = int((Nf * f / fe - 1) / 2)
    hf = (1 / Nf) * np.sin(np.pi * nf * k / Nf) / (np.sin(np.pi * nf / Nf) + 1e-20)
    hf[int(Nf / 2)] = k / Nf
    wf = np.hanning(Nf)
    hwf = hf * wf
    env = np.convolve(hwf, abs(x))
    sol = synthesis_note(N, amp, ph, 392.0, env[0:160000])
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.subplot(2, 1, 2)
    plt.plot(sol)
    plt.show()
    sf.write("sol.wav", sol, fe)
    pass
