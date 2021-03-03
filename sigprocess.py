#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:06:08 2021

@author: darwin
"""

import wave
import numpy

# Read file to get buffer                                                                                               
ifile = wave.open("sine14k.wav")
samples = ifile.getnframes()
audio = ifile.readframes(samples)

# Convert buffer to float32 using NumPy                                                                                 
audio_as_np_int16 = numpy.frombuffer(audio, dtype=numpy.int16)
audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)


# Normalise float32 array so that values are between -1.0 and +1.0                                                      
max_int16 = 2**15
audio_normalised = audio_as_np_float32 / max_int16

#%%

from scipy.io.wavfile import read

a = read("noise.wav")

fs = a[0] # samples per second
arr = numpy.array(a[1],dtype=float)

chunk_size = 0.02 # seconds
 
samples_in_chunk = int(chunk_size * fs)

chunks = []

for i in range(0, arr.shape[0], samples_in_chunk):
    chunks.append(arr[i:i + samples_in_chunk])

spectrum = [
    ]
for chunk in chunks :
    spectrum.append(())

t = numpy.arange(0, arr.shape[0])

from matplotlib import pyplot as plt

plt.title("Matplotlib demo") 
plt.xlabel("t") 
plt.ylabel("y") 
plt.plot(t[: samples_in_chunk], arr[:samples_in_chunk])
plt.show()



#%%

import numpy as np
from matplotlib import pyplot as plt

SAMPLE_RATE = 16000  # Hertz
DURATION = 5  # Seconds

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

# Generate a 2 hertz sine wave that lasts for 5 seconds
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)


noise_tone = noise_tone * 0.3

mixed_tone = nice_tone + noise_tone

normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * (2 ** 15 - 1))

from scipy.io.wavfile import read
audio = read("noise.wav")
fs = audio[0] # samples per second
audio = numpy.array(audio[1],dtype=float)

plt.plot(normalized_tone[:100])
plt.show()

from scipy.io import wavfile
wavfile.write("mix.wav", SAMPLE_RATE, normalized_tone)



from scipy.fft import fft, fftfreq, rfft, rfftfreq

# Number of samples in normalized_tone
N = audio.shape[0]

yf = rfft(audio)
xf = rfftfreq(N, 1 / fs)


plt.plot(xf, np.abs(yf))
plt.show()


















