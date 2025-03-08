import queue
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

class AudioFFT():
    def __init__(self):
        self.file = None
    
    def read_file(self):
        self.input_data, self.channels = sf.read(self.file)
        self.data = self.input_data[0 : len(self.input_data)]
        return self
    
    def set_file(self, filepath: str):
        self.file = filepath
        self.read_file()
        return self

    def plot_data(self):
        pd.DataFrame(self.data).plot(figsize=(16,8))

    def slice_data(self, start: int = None, end: int = None):
        if start == None:
            start = 0
        if end == None:
            end = len(self.input_data)
        self.data = self.input_data[int(start) : int(end)]

    def perform_fft(self, channel: int = 0):
        self.fdata = np.fft.rfft(self.data[:, channel])
        self.fdata_trimmed = self.fdata

    def plot_fdata(self):
        pd.DataFrame(self.fdata_trimmed).plot(figsize=(16,8))

    def trim_fdata(self, tol: float=5):
        for i, point in reversed(list(enumerate(self.fdata))):
            if abs(point) > tol:
                print(i, point)
                break
        self.fdata_trimmed = self.fdata[:i]


    def perform_ifft(self, use_trimmed: bool = False):
        if use_trimmed:
            fdata = self.fdata_trimmed
        else:
            fdata = self.fdata
        self.idata = np.fft.irfft(fdata)

    def plot_idata(self):
        pd.DataFrame(self.idata).plot(figsize=(16,8))
    

if __name__ == "__main__":
    sns.set_style("whitegrid")
    audio_fft = AudioFFT()
    fig, ax = plt.subplots(figsize=(8,16))
    #audio_fft.set_file(r'2025-2_3-audio\assets\315705__spitefuloctopus__acoustic-guitar-d-major-chord-short.wav')
    audio_fft.set_file(r"2025-2_3-audio\assets\123033__cgeffex__guitar_strings_take1.wav")
    audio_fft.slice_data(0,0.1e6)
    #audio_fft.plot_data()
    audio_fft.perform_fft()
    audio_fft.trim_fdata()
    audio_fft.plot_fdata()
    audio_fft.perform_ifft()
    #audio_fft.plot_idata()
    plt.show()



    