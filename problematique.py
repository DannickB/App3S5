import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
import wave
import struct
import scipy.io.wavfile as wav
import sounddevice as sd

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
    fund = np.argmax(abs(X))
    peak_ids, _ = ss.find_peaks(np.abs(X[0:int(len(X)/2)]), distance=200)
    peaks=[]
    for peak in peak_ids:
        peaks.append(X[peak])
    peaks = sorted(peaks, reverse=True, key=abs)
    for i in range(0, 32):
        amp.append(abs(peaks[i]))
        ph.append(np.angle(peaks[i]))
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


def synthesis_note(N, amps, ph, freq, env, fe, signal):
    t = np.linspace(0, (N/fe), N)
    x = []
    for dt in t:
        val = 0
        for i in range(len(amps)):
            val += amps[i] * np.sin(2*np.pi*freq*i*dt - ph[i])
        x.append(val)
    x = np.multiply(x, env)
    x *= max(np.abs(signal) / max(np.abs(x)))
    return x

if __name__ == "__main__":
    # signal, fe = sf.read('note_guitare_lad.wav')
    fe, signal = wav.read('note_guitare_lad.wav')
    frequency_map = frequencyMap()
    f = frequency_map["LA#"]
    N = len(signal)
    X = np.fft.fft(signal, fe)
    amp, ph = extract_params(X[0:int(fe/2)])

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

    #Creating desired notes
    sol = synthesis_note(N, amp, ph, frequency_map["LA#"], env[0:N], fe, signal)
    mi = synthesis_note(N, amp, ph, frequency_map["RE#"], env[0:N], fe, signal) #Re# et Mi bemole sont la meme
    fa = synthesis_note(N, amp, ph, frequency_map["FA"], env[0:N], fe, signal)
    re = synthesis_note(N, amp, ph, frequency_map["RE"], env[0:N], fe, signal)
    silence = np.zeros(int(fe/2))

    #Stripping notes because the full duration is too long for the melody
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
    # plt.show()
    sd.play(np.int16(np.real(symphony)), fe)
    wav.write("symphony.wav", fe, np.int16(np.real(symphony)))

