import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def extract_params(X, freqs):
    amp = []
    ph = []
    id_fund = np.argmax(abs(X))
    for i in range(0, 33):
        amp.append(abs(X[id_fund * i]))
        ph.append(np.angle(X[id_fund * i]))
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


def synthesis_note(N, amps, ph, freq, env, fe):
    t = np.linspace(0, (N/fe), N)
    x = []
    for dt in t:
        val = 0
        for i in range(33):
            val += amps[i] * np.sin(2*np.pi*freq*i*dt + ph[i])
        x.append(val)
    return x * env


if __name__ == "__main__":
    signal, fe = sf.read('note_guitare_lad.wav')
    f = 466
    N = len(signal)
    w = np.hamming(N)
    signalW = signal * w
    X = np.fft.fft(signalW, N)
    freq_arr = np.fft.fftfreq(N)
    amp, ph = extract_params(X, freq_arr)


    filter_freq = np.pi / 1000
    target_gain = np.sqrt(2) / 2
    Nf = find_order(filter_freq, target_gain)
    nf = np.arange(-Nf / 2, Nf / 2)
    k = int((Nf * f / fe - 1) / 2)
    hf = (1 / Nf) * np.sin(np.pi * nf * k / Nf) / (np.sin(np.pi * nf / Nf) + 1e-20)
    hf[int(Nf / 2)] = k / Nf
    wf = np.hanning(Nf)
    hwf = hf * wf
    env = np.convolve(hwf, abs(signal))
    env_max = env[np.argmax(env)]
    env = env/env_max

    sol = synthesis_note(N, amp, ph, 392.0, env[0:160000], fe)
    a1 = plt.subplot(3, 1, 1)
    a1.set_title("Base sample")
    a1.plot(signal)
    a2 = plt.subplot(3, 1, 2)
    a2.set_title("new note")
    a2.plot(sol)
    a3 = plt.subplot(3, 1, 3)
    a3.set_title("envloppe")
    a3.plot(env)
    plt.show()
    sf.write("sol.wav", sol, fe)
    pass
