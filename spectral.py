#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:37:23 2021

@author: darwin
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
from scipy.fft import rfft, rfftfreq


def getAudioAsNpArray(file) :
    audio = read(file)
    sampling_rate = audio[0] # samples per second
    print(file, "sampling rate:", sampling_rate)
    audio_array = np.array(audio[1], dtype=float)
    return sampling_rate, audio_array

def getAudioChunks(chunk_duration, sampling_rate, audioArray) :
    n_chunks = len(audioArray) //  (chunk_duration * sampling_rate)
    return np.array_split(audioArray, n_chunks)

def getChunkSpectrum(sampling_rate, audio_chunk):
    N = len(audio_chunk)
    yf = rfft(audio_chunk)
    xf = rfftfreq(N, 1 / sampling_rate)
    return xf, yf

def getMeanSpectrum(sampling_rate, audio_chunks):
    xf0, yf0 = getChunkSpectrum(sampling_rate, audio_chunks[2])
    spectrum_mat = yf0.reshape(1, len(yf0))
    
    for chunk in audio_chunks[1:-1] :
        xf, yf = getChunkSpectrum(sampling_rate, chunk)
        spectrum_mat = np.concatenate((spectrum_mat, np.abs(yf).reshape(1, len(yf))), axis=0)
    
    spectrum_mean = np.mean(spectrum_mat, axis=0)
    return xf0, spectrum_mean

def plotSpectrum(xf, yf, name) :
    plt.title(name)
    plt.plot(xf, np.abs(yf))
    plt.show()
    
def plotSpectrums(spectrums) :
    print(plt.style.available)
    plt.style.use('fivethirtyeight')
    plt.title("spectrums")
    plt.xlabel("frequencies")
    plt.ylabel("amplitude")
    for spectrum in spectrums :
        plt.plot(spectrum["xf"], np.abs(spectrum["yf"]), linewidth=1, label=spectrum["name"])
        
    plt.legend()
    plt.tight_layout()
    plt.savefig("spec.png", dpi=500)
    plt.show()

def spectralAnalysisPlot(audio_files) :
    audio_signals = []
    for f in audio_files :
        fs, arr = getAudioAsNpArray(f)
        audio_signals.append({"array": arr, "fs": fs, "name": f})

    chunk_size = 0.02  # seconds
    audio_spectrum = []    
    for signal in audio_signals :
        chunks = getAudioChunks(chunk_size, signal["fs"], signal["array"])
        mxf, myf = getMeanSpectrum(fs, chunks)
        audio_spectrum.append({"xf": mxf, "yf": myf, "name": signal["name"]})
    
    # for spectrum in audio_spectrum :
    #     plotSpectrum(spectrum["xf"], spectrum["yf"], spectrum["name"])
    
    plotSpectrums(audio_spectrum)
#%%

# audio_file = "aaa.wav"

# fs, arr = getAudioAsNpArray(audio_file)
# chunks = getAudioChunks(0.02, fs, arr)

# xf, yf = getChunkSpectrum(fs, arr)
# mxf, myf = getMeanSpectrum(fs, chunks)

# plotSpectrum(xf, yf)
# plotSpectrum(mxf, myf)


#%%

audio_files = ["aaa.wav", "eee.wav", "ooo.wav", "noise.wav"]

spectralAnalysisPlot(audio_files)










