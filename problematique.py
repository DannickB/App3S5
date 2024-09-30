import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import wave
import struct
from scipy.io.wavfile import write

def frequencyMap():
    freqs = \
    { "DO":261.6,
      "DO#":277.2,
      "RE":293.7,
      "RE#":311.1,
      "MI":329.6,
      "FA":349.2,
      "FA#":370.0,
      "SOL":392.0,
      "SO#":415.3,
      "LA":440.0,
      "LA#":466.2,
      "SI":493.9
    }
    return freqs

def extract_params(X):
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
    return np.multiply(x, env)

def save_audio(audio, sampleRate, filename):
    out = wave.open(filename, 'wb')
    out.setnchannels(1)
    out.setframerate(sampleRate)
    out.setsampwidth(2)
    for frame in audio:
        data = struct.pack('<h', int(frame*8))
        out.writeframes(data)
    out.close()

if __name__ == "__main__":
    signal, fe = sf.read('note_guitare_lad.wav')
    frequency_map = frequencyMap()
    f = frequency_map["LA#"]
    N = len(signal)
    w = np.hamming(N)
    X = np.fft.fft(signal, N)
    Xw = np.multiply(X, w)
    amp, ph = extract_params(Xw)

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


    sol = synthesis_note(N, amp, ph, frequency_map["SOL"], env[0:N], fe)
    mi = synthesis_note(N, amp, ph, frequency_map["RE#"], env[0:N], fe) #Re# et Mi bemole sont la meme
    fa = synthesis_note(N, amp, ph, frequency_map["FA"], env[0:N], fe)
    re = synthesis_note(N, amp, ph, frequency_map["RE"], env[0:N], fe)
    silence = np.zeros(int(fe/2))

    t_quarter = int(N/4)
    t_demi = int(N/2)
    symphony = np.concat((sol[0:t_quarter], sol[0:t_quarter], sol[0:t_quarter], mi[0:t_demi], silence, fa[0:t_quarter], fa[0:t_quarter], fa[0:t_quarter], re))

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
    save_audio(symphony.tolist(), fe, "symphony.wav")

