###################################################
# APP3 S5
# Resolution de la problematique
# labg0902 - bild2707
###################################################

import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def frequencyMap():
    """
    Make a map of every fundamental frequency (in Hz) for every notes
    :return: Dictionary of frequencies
    """
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

def plot_harmonics(harm , X):
    n = np.zeros(len(X))
    for i in range(32):
        n[harm[i][1]] = abs(harm[i][0])
    plt.figure("Harmonics")
    plt.plot(n)
    plt.title("Harmonics")

def extract_params(X):
    """
    Extract parameters from an audio signal
    :param X: Input audio signal in frequency
    :return: List of parameters (amplitude and phase) for the first 32 harmonics
    """
    amp = []
    ph = []
    fund = np.argmax(abs(X))
    # Finds every local peaks by region of half the fundamental frequency
    peak_ids, _ = ss.find_peaks(np.abs(X[0:int(len(X)/2)]), distance=fund/2)
    peaks=[]
    #Order peaks by magnitude to make sure to keep only real peaks
    for peak in peak_ids:
        peaks.append([X[peak], peak])
    peaks = sorted(peaks, reverse=True, key=lambda x:abs(x[0]))
    plot_harmonics(peaks, X)
    #Save the first 32 harmonics in arrays
    for i in range(0, 32):
        amp.append(abs(peaks[i][0]))
        ph.append(np.angle(peaks[i][0]))
    return amp, ph


def find_order(freq, target):
    """
    Find order N for an FIR filter by trying every value of N until it arrive to the target gain
    :param freq: Frequency for specific gain
    :param target: Target gain
    :return: Order N
    """
    N = 0
    Xn = 1 # result of function sum(1/n * exp(-j*w*n)
    T = 0
    while Xn - target > 0: # Gain will go down the greater N is, we want to stop once we pass the target
        N += 1
        T += np.exp(-1j * freq * N) # exp(-j*w*n) part (Increment since it's a sum)
        Xn = np.abs(T / N) # 1/N part
    return N


def synthesis_note(N, amps, ph, freq, env, sr, signal):
    """
    Recreate sound from params
    :param N: total number of samples
    :param amps: Array of amplitudes
    :param ph: Array of phases
    :param freq: Fundamental frequency
    :param env: Envelope
    :param sr: Sample rate
    :param signal: Input signal
    :return: Synthesised signal
    """
    t = np.linspace(0, (N/sr), N)
    x = []
    #For every n points, compute the sum of every harmonic and append the results to the output signal
    for dt in t:
        val = 0
        for i in range(len(amps)):
            val += amps[i] * np.sin(2*np.pi*freq*i*dt - ph[i])
        x.append(val)
    x = np.multiply(x, env)
    x *= max(np.abs(signal)) / max(np.abs(x))
    return x

if __name__ == "__main__":
    #reading signal
    sr, signal = wav.read('note_guitare_lad.wav')
    frequency_map = frequencyMap()
    f = frequency_map["LA#"]
    start = 7650  # Starting frame of audio
    N = len(signal)
    X = np.fft.fft(signal, sr)
    amp, ph = extract_params(X[0:int(sr/2)])
    plt.figure("Fft du signal d'entrée")
    plt.title("Fft du signal d'entrée")
    plt.plot(np.fft.fftshift(np.abs(X)))

    #Envelope segment
    filter_freq = np.pi / 1000 # Frequency of known gain
    target_gain = np.sqrt(2) / 2 # Wanted gain at said frequency
    Nf = find_order(filter_freq, target_gain) #Order of low pass filter
    nf = np.arange(-Nf / 2, Nf / 2)
    k = int((2 * Nf * filter_freq / sr ) + 1) # finding k from N since we know sample rate and frequency wanted
    hf = (1 / Nf) * np.sin(np.pi * nf * k / Nf) / (np.sin(np.pi * nf / Nf) + 1e-20)
    hf[int(Nf / 2)] = k / Nf
    w = np.hanning(Nf)
    hwf = hf * w
    plt.figure("Filtre passe bas")
    plt.title("Filtre passe bas")
    plt.plot(hwf)
    env = np.convolve(hwf, abs(signal))
    env_max = env[np.argmax(env)]
    env = env/env_max

    #Creating desired notes
    sol = synthesis_note(N, amp, ph, frequency_map["SOL"], env[0:N], sr, signal)
    mi = synthesis_note(N, amp, ph, frequency_map["RE#"], env[0:N], sr, signal) #Re# et Mi bemole sont la meme
    fa = synthesis_note(N, amp, ph, frequency_map["FA"], env[0:N], sr, signal)
    re = synthesis_note(N, amp, ph, frequency_map["RE"], env[0:N], sr, signal)
    silence = np.zeros(int(sr/2))

    #Stripping notes because the full duration is too long for the melody
    t_quarter = int(N/4)
    t_demi = int(N/2)
    symphony = np.concat((sol[0:t_quarter], sol[start:t_quarter], sol[start:t_quarter], mi[start:t_demi], silence, fa[start:t_quarter], fa[start:t_quarter], fa[start:t_quarter], re[start:N]))
    plt.figure("Formes d'onde")
    a1 = plt.subplot(3, 1, 1)
    a1.set_title("Base sample")
    a1.plot(signal)
    a2 = plt.subplot(3, 1, 2)
    a2.set_title("symphony")
    a2.plot(sol)
    a3 = plt.subplot(3, 1, 3)
    a3.set_title("envelope")
    a3.plot(env)
    plt.show()

    wav.write("symphony.wav", sr, np.int16(np.real(symphony)))

