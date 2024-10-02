###################################################
# APP3 S5
# Resolution de la problematique
# labg0902 - bild2707
###################################################

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import scipy as sc
from guitare import *
from numpy.lib.format import write_array_header_1_0

# Settings
x, fe = sf.read('note_basson_plus_sinus_1000_hz.wav')
len = len(x)
N = 6000
fc = 1000
fw = 40
w0 = (2 * np.pi * fc) / fe
w1 = (2 * np.pi * fw) / fe
n = np.arange(-N/2, N/2)
m = (N * fw) / fe
k = 2 * m + 1

# Filtre passe bas correspondant
hf = np.sin(np.pi * k * n / 6000) / ( N * np.sin(np.pi * n / 6000) + 1e-20)
# Et sa transformation en coupe bande
diracht = np.zeros(N)
diracht[3000] = 1
# Reponse impulsionnelle du filtre coupe bande
hfc = diracht - 2 * hf * np.cos(w0 * n)

# Fenetrage
window = np.hanning(N)
h = hfc * window

# Convolution pour obtenir le signal filtre du 1000 Hz
y1 = np.convolve(h,x)

H = sc.fft.fftshift(sc.fft.fft(h))
X = sc.fft.fft(x)

# Calculs de l'enveloppe grace au filtre passe-bas de la guitare
filter_freq = np.pi / 1000 # Frequency of known gain
target_gain = np.sqrt(2) / 2 # Wanted gain at said frequency
Nf = find_order(filter_freq, target_gain) #Order of low pass filter
nf = np.arange(-Nf / 2, Nf / 2)
k = int((2 * Nf * filter_freq / fe) + 1) # finding k from N since we know sample rate and frequency wanted
hf = (1 / Nf) * np.sin(np.pi * nf * k / Nf) / (np.sin(np.pi * nf / Nf) + 1e-20)
hf[int(Nf / 2)] = k / Nf
env = np.convolve(hf, abs(y1))
env_max = env[np.argmax(env)]
env = env / env_max
Y = np.fft.fft(y1, fe)

# Parametrages
amp, ph = extract_params(X[0:int(fe/2)])

# Synthetisation du nouveau signal (sera tres inadequant car on reutilise le filtre passe bas de la guitare)
fad = synthesis_note(len, amp, ph, 466.2, env[0:len], fe, y1)

# Plotting
plt.figure("specs")
plt.subplot(3,2,1)
plt.plot(n, h)
plt.subplot(3,2,2)
plt.xlim(900, 1100)
plt.plot(n*fe/N,np.abs(H))
plt.subplot(3,2,3)
plt.plot(x)
plt.subplot(3,2,4)
plt.plot(np.abs(np.abs(X)))
plt.xlim(0, 44100)
plt.subplot(3,2,5)
plt.plot(np.abs(y1))
plt.subplot(3,2,6)
plt.xlim(0,22050)
plt.plot(np.arange(-fe/2,fe/2),20*np.log10(np.fft.fftshift(np.abs(Y))))
plt.tight_layout()
plt.figure("Synthesised")
plt.plot(fad)
plt.show()

sf.write("basson.wav",y1,fe)
sf.write("basson_synth.wav",fad,fe)