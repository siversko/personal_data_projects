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
        self.input_data, self.samplerate = sf.read(self.file)
        self.data = self.input_data[0 : len(self.input_data)]
        self.channels = len(self.input_data[0])
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

    def set_data(self, data):
        self.data = data
        return self
    
    def get_data(self):
        ''' Returns soundwave data'''
        return self.data
    
    def set_times(self, data):
        self.times = np.linspace(0, 
                                 len(data[:,0]) / self.samplerate, # samples / (samples /sec) => secs
                                 len(data[:,0]))
        print(f'total time {self.times[-1]}s')
        return self

    def set_fdata(self):
        ''' Returns fourier transformed data'''
        fdata = []
        data = self.data.transpose()
        for channel in range(self.channels):
            fdata.append(np.fft.rfft(self.data[:, channel]))
        self.fdata = np.array(fdata).transpose()
        return self

    def get_fdata(self):
        return self.fdata
    
    def set_frequencies(self):
        self.frequenies = np.fft.rfftfreq(len(self.data[:, 0]), 1/self.samplerate)
        self.dominant_freq_dict = self.find_dominant_frequencies(tol = 0.2)
        return self

    def set_idata(self, fdata):
        fdata = fdata.transpose()
        idata = []
        for channel in range(self.channels):
            idata.append(np.fft.irfft(fdata[channel]))
        self.idata = np.array(idata).transpose()
        return self
    
    def get_idata(self):
        return self.idata
    
    def set_fidata(self):
        return self.set_fdata() \
                   .set_frequencies() \
                   .set_idata(self.fdata) \
                   .set_times(self.idata) \

    def signal_plot(self, data, ax):
        self.set_times(data)
        lines = ax.plot(self.times, data)
        ax.set(xlabel='time (s)',
               ylabel='Amplitude')
        ax.legend(lines, list(range(len(data[0]))), title='Channel')
        return ax
    
    def find_dominant_frequencies(self, tol: float = 0.1):
        dom_freq_dict = {}
        channel = 0
        abs_fdata = np.abs(self.fdata.transpose())
        for channel_data in abs_fdata:
            dom_freqs = []
            maximum_amp = max(np.abs(channel_data))
            print('maximum_amp', maximum_amp)
            for i, amp in enumerate(channel_data):
                if i == len(channel_data):
                    break
                # check sufficient intensity
                if amp > maximum_amp*tol:
                    #check local maximum
                    if (amp > channel_data[i-1]) and (amp > channel_data[i+1]):
                        dom_freqs.append((i, amp))
            dom_freq_dict[channel] = dom_freqs
            channel +=1
        self.dom_freq_dict = dom_freq_dict
        return dom_freq_dict

    def dampen_frequency(self, fdata, target_freq: float, damp: float = 0.8, band_size: int = 10):
        dataT = fdata.T.copy() # shallow copy?
        for channel in dataT:
            for i, amp in enumerate(channel):
                if np.round(self.frequenies[i], 2) == target_freq:
                    # print(f'index {i}, {amp}, at {self.frequenies[i]}')
                    # print(f'damped {i}, {amp*(1-damp)}, at {self.frequenies[i]}')
                    break

            for j in range(i-band_size, i+band_size):
                if j < len(channel):
                    channel[j] *= (1-damp)
        #return np.zeros(dataT.T.shape)
        return dataT.transpose()
    
    def dampen_frequencies(self, target_freqs: list[float], damp:float = 0.8, band_size:int = 10):
        dampened_data = self.get_fdata()
        for freq in target_freqs:
            dampened_data = self.dampen_frequency(dampened_data, freq, damp, band_size)
        return dampened_data

    def frequency_plot(self, fdata, ax, pad: int = 10):
        #self.set_frequencies()
        #print(fdata.shape)
        #print(self.frequenies.shape)
        lines = ax.plot(self.frequenies, np.abs(fdata))
        #lines = ax.plot(np.abs(fdata))
        ax.set(xlabel='frequency (Hz)',
               ylabel='Amplitude')
        ax.legend(lines, list(range(len(fdata[0]))), title='Channel')
        freq_lim = 0
        for channel in self.dominant_freq_dict:
            #print(dominant_freq_dict[channel][-1])
            if freq_lim < self.dominant_freq_dict[channel][-1][0]:
                freq_lim = self.dominant_freq_dict[channel][-1][0]
        ax.set_xlim(left = max(0, -pad), right=self.frequenies[freq_lim+pad])
        colors = ['black', 'red']
        for channel in self.dominant_freq_dict:
            #print(channel)
            for index_amp in self.dominant_freq_dict[channel]:
                #print(index_amp[0])
                #print(self.frequenies[index_amp[0]], np.abs(fdata)[index_amp[0]])
                ax.annotate(np.round(self.frequenies[index_amp[0]],2), xy=(self.frequenies[index_amp[0]], np.abs(fdata)[index_amp[0]][channel]), color=colors[channel])
        return ax
    
    def play_sound(self, data):
        sd.play(data, self.samplerate)

### Functions ###

def plot_data(data):
    pd.DataFrame(data).plot(figsize=(16,8))

def cosine_similarity(data1, data2):
    data1T = data1.transpose()
    data2T = data2.transpose()
    similarities = []
    for channel in range(len(data1T)):
        similarities.append(1 - sp.spatial.distance.cosine(data1T[channel], data2T[channel]))
    return similarities
    #return sp.spatial.distance.cosine(data1, data2)

def slice_data(data, f_index, b_index, pad: int):
    if f_index - pad < 0:
        f_index = 0
    if b_index + 10 > len(data):
        b_index = len(data)
    #print(f'slicing, from index {f_index}, to index {b_index}')
    return data[f_index : b_index]

def slice_time(data, samplerate, f_time, b_time, pad: int):
    f_index = int(f_time * samplerate)
    b_index = int(b_time * samplerate)
    return slice_data(data, f_index, b_index, pad=10)

def find_trim_indexes(dataT, direction: str = 'fb', tol: float = 0.05):
    max_amplitudes = []
    f_index, b_index = 0, -1
    for channel_data in dataT:
        #print(channel_data)
        max_amplitude = max(channel_data)
        max_amplitudes.append(max_amplitude)
        if 'f' in direction:
            for i, point in enumerate(channel_data):
                #print(i, point)
                if point >= max_amplitude*tol:
                    #print(f'index {i}, value {point}, amplitude tolerance {max_amplitude*tol}')
                    if i < f_index or f_index == 0:
                        f_index = i
                    break
        else:
            f_index = 0
        if 'b' in direction:
            for i, point in reversed(list(enumerate(channel_data))):
                #print(i,point)
                if point >= max_amplitude*tol:
                    #print(f'index {i}, value {point}, amplitude tolerance {max_amplitude*tol}')
                    if i > b_index:
                        b_index = i
                    break
        else:
            b_index = -1
    return f_index, b_index

def trim_data(data, direction: str = 'fb', tol: float = 0.05, pad: int = 10):
    #print(data.shape)
    dataT = data.transpose()
    data_trimmed = []
    f_index, b_index = find_trim_indexes(dataT)
    for channel_data in dataT:
        data_trimmed.append(slice_data(channel_data, f_index, b_index, pad))
    data_trimmed = np.array(data_trimmed)
    #print(data_trimmed.shape)
    return data_trimmed.transpose()

if __name__ == "__main__":
    sns.set_style("whitegrid")
    audio_fft = AudioFFT()
    #fig, ax = plt.subplots(figsize=(8,16))
    #audio_fft.set_file(r'2025-2_3-audio\assets\315705__spitefuloctopus__acoustic-guitar-d-major-chord-short.wav')
    audio_fft.set_file(r"2025-2_3-audio\assets\123033__cgeffex__guitar_strings_take1.wav")
    # audio_fft.slice_data(0,0.1e6)
    
    # audio_fft.plot_data()
    # audio_fft.perform_fft()
    # audio_fft.trim_fdata()
    # #audio_fft.plot_fdata()
    # audio_fft.perform_ifft()
    # audio_fft.plot_idata()
    #audio_fft.set_fdata().set_idata()
    data = audio_fft.get_data()
    #idata = audio_fft.get_idata()
    #fdata = audio_fft.get_fdata()
    #print(cosine_similarity(data,idata))
    #plot_data(data)
    #plot_data(fdata)
    sliced = slice_time(data, audio_fft.samplerate, 0, 3, 10)
    trim = trim_data(sliced)
    #plot_data(trim)
    audio_fft.set_data(trim)
    #audio_fft.set_times()
    audio_fft.set_fidata()
    #plot_data(audio_fft.get_fdata())
    #plot_data(audio_fft.get_idata())
    audio_fft.find_dominant_frequencies()
    # fig,axs = plt.subplots(2,1, figsize=(16,8))
    # # print(axs)
    # audio_fft.signal_plot(trim, axs[0])
    # audio_fft.frequency_plot(audio_fft.get_fdata(), axs[1])
    # # #plt.legend()
    # plt.show()
    fig,axs = plt.subplots(4,1, figsize=(16,8))
    audio_fft.signal_plot(trim, axs[0])
    audio_fft.frequency_plot(audio_fft.get_fdata(), axs[1])
    audio_fft.frequency_plot(audio_fft.dampen_frequency(audio_fft.get_fdata(), 222.95), axs[1])
    audio_fft.frequency_plot(audio_fft.dampen_frequencies([164.16, 82.08, 243.37, 247.88, 411.22], damp=0.99), axs[2])
    audio_fft.set_idata(audio_fft.dampen_frequencies([164.16, 82.08, 243.37, 247.88, 411.22], damp=0.99))
    audio_fft.signal_plot(audio_fft.get_idata(), axs[3])
    audio_fft.play_sound(audio_fft.get_data())
    #sd.wait()
    audio_fft.play_sound(audio_fft.get_idata())
    #sd.wait()
    print(cosine_similarity(audio_fft.get_data(), audio_fft.get_idata()))
    plt.show()