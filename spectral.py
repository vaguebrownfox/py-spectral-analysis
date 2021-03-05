#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:37:23 2021

@author: darwin
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.io.wavfile import read
from scipy.fft import rfft, rfftfreq
import os


def getAudioAsNpArray(file):
    print("filename", file)
    audio = read(file)
    sampling_rate = audio[0]  # samples per second
    print(file, "sampling rate:", sampling_rate)
    audio_array = np.array(audio[1], dtype=float)
    return sampling_rate, audio_array


def getAudioChunks(chunk_duration, sampling_rate, audioArray):
    n_chunks = len(audioArray) // (chunk_duration * sampling_rate)
    return np.array_split(audioArray, n_chunks)


def getChunkSpectrum(sampling_rate, audio_chunk):
    N = len(audio_chunk)
    yf = rfft(audio_chunk)
    xf = rfftfreq(N, 1 / sampling_rate)
    return xf, yf


def getMeanSpectrum(sampling_rate, audio_chunks):
    xf0, yf0 = getChunkSpectrum(sampling_rate, audio_chunks[2])
    spectrum_mat = yf0.reshape(1, len(yf0))

    for chunk in audio_chunks[1:-1]:
        xf, yf = getChunkSpectrum(sampling_rate, chunk)
        spectrum_mat = np.concatenate(
            (spectrum_mat, np.abs(yf).reshape(1, len(yf))), axis=0
        )

    spectrum_mean = np.mean(spectrum_mat, axis=0)
    return xf0, spectrum_mean


def plotSpectrum(xf, yf, name):
    plt.title(name)
    plt.plot(xf, np.abs(yf))
    plt.show()


fontP = FontProperties()
fontP.set_size("xx-small")


def plotSpectrums(spectrums, title):

    plt.figure()
    plt.style.use("seaborn-paper")
    plt.title(title)
    plt.xlabel("frequencies")
    plt.ylabel("amplitude")
    # 20 * np.log10(np.abs(spectrum["yf"]))
    for spectrum in spectrums:
        plt.plot(
            spectrum["xf"],
            20 * np.log10(np.abs(spectrum["yf"])),
            linewidth=1,
            label=spectrum["name"],
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop=fontP)
    plt.tight_layout()
    plt.savefig(f"plots/{title}.png", dpi=500)
    # plt.show()


def spectralAnalysisPlot(audio_file_paths, audio_files, title):
    audio_signals = []
    for i, f in enumerate(audio_file_paths, start=0):
        fs, arr = getAudioAsNpArray(f)
        audio_signals.append({"array": arr, "fs": fs, "name": audio_files[i]})
        i = +1

    chunk_size = 0.02  # seconds
    audio_spectrum = []
    for signal in audio_signals:
        chunks = getAudioChunks(chunk_size, signal["fs"], signal["array"])
        mxf, myf = getMeanSpectrum(fs, chunks)
        audio_spectrum.append({"xf": mxf, "yf": myf, "name": signal["name"]})

    # for spectrum in audio_spectrum :
    #     plotSpectrum(spectrum["xf"], spectrum["yf"], spectrum["name"])

    plotSpectrums(audio_spectrum, title)


#%%

# audio_file = "aaa.wav"

# fs, arr = getAudioAsNpArray(audio_file)
# chunks = getAudioChunks(0.02, fs, arr)

# xf, yf = getChunkSpectrum(fs, arr)
# mxf, myf = getMeanSpectrum(fs, chunks)

# plotSpectrum(xf, yf)
# plotSpectrum(mxf, myf)


#%%


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


audio_files = ["aaa/ooo.wav", "aaa/mmm.wav"]

aaa = filter(lambda x: (".wav" in x), list(absoluteFilePaths("./AAA")))
eee = filter(lambda x: (".wav" in x), list(absoluteFilePaths("./EEE")))
uuu = filter(lambda x: (".wav" in x), list(absoluteFilePaths("./UUU")))
white = filter(lambda x: (".wav" in x), list(absoluteFilePaths("./WhiteNoise")))
# eee = list(absoluteFilePaths("./EEE"))
# uuu = list(absoluteFilePaths("./UUU"))
# white = list(absoluteFilePaths("./WhiteNoise"))


aaa_filenames = os.listdir("./AAA")
aaa_filenames.remove(".DS_Store")

eee_filenames = os.listdir("./EEE")
eee_filenames.remove(".DS_Store")

uuu_filenames = os.listdir("./UUU")
uuu_filenames.remove(".DS_Store")

white_filenames = os.listdir("./WhiteNoise")
white_filenames.remove(".DS_Store")

# white_filenames = os.listdir("./WhiteNoise")
# print(os.listdir("./aaa"))

spectralAnalysisPlot(white, white_filenames, "white")
spectralAnalysisPlot(aaa, aaa_filenames, "aaa")
spectralAnalysisPlot(eee, eee_filenames, "eee")
spectralAnalysisPlot(uuu, uuu_filenames, "uuu")