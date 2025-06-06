import os, queue
import functools
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble
import sklearn.preprocessing
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sklearn

import audio_fft
import chord_classifier
import chroma

class ReferenceSounds():
    ''' Deprecated '''
    def __init__(self, reference_dir: str):
        self.reference_dir = os.path.abspath(reference_dir)
        print(reference_dir)
        self.reference_frequencies = dict()
        a_fft = audio_fft.AudioFFT()
        a_fft.samplerate = sd.default.samplerate
        a_fft.channels = sd.default.channels
        self.sound_detected = False
        self.counter = Counter()
        self.chord_clf = chord_classifier.get_chord_classifier()
        
        for reference_file in os.listdir(self.reference_dir):
            #print(reference_dir + '\\' + reference_file)
            a_fft.set_file(reference_dir + '\\' + reference_file) #files already trimmed
            a_fft.set_fdata()
            a_fft.set_frequencies()
            #a_fft.find_dominant_frequencies()

            self.reference_frequencies[reference_file.split('.')[0]] = a_fft.get_fdata()
        #print(self.reference_frequencies.keys())
        self.chord_scores = {chord : [] for chord in self.reference_frequencies.keys()}

    def cut_to_size(self):
        shortest_chord_array_length = np.inf
        for chord in self.reference_frequencies:
            if self.reference_frequencies[chord].shape[0] < shortest_chord_array_length:
                shortest_chord_array_length = self.reference_frequencies[chord].shape[0] -1
        for chord in self.reference_frequencies:
            self.reference_frequencies[chord] = self.reference_frequencies[chord][:shortest_chord_array_length, :]
        return self
    
    def score_freq(self, freq: np.ndarray, reference: np.ndarray):
        #return audio_fft.cosine_similarity(freq, reference)
        # scores = []
        # for channel, _ in enumerate(reference.T):
        #     scores.append(np.correlate(np.abs(freq.T[channel]), np.abs(reference.T[channel])))
        # return scores
        scores = []
        for channel in freq.T:
            padded_channel = np.pad(np.abs(channel).reshape(1, -1), (0, 24410-len(channel)), mode='constant') # 
            print(self.chord_clf.predict(padded_channel), len(self.chord_clf.predict(padded_channel)))
            print(self.chord_clf.predict_proba(padded_channel))
            scores.append(np.max(self.chord_clf.predict_proba(padded_channel)))
        print(scores)
        return scores


    def classify_freq(self, freq):
        scores = dict()
        top_score = 0
        top_chord = ''
        for chord in self.reference_frequencies:
            scores[chord] = self.score_freq(freq, self.reference_frequencies[chord])
            score = np.mean(scores[chord])
            self.chord_scores[chord].append(score)
            if top_score < score:
                top_score = score
                top_chord = chord
        self.counter[top_chord] += 1
        for chord in scores:
            print(chord, np.mean(scores[chord]), len(scores[chord]))
        return top_chord, top_score
    
    def classify_freq2(self, freq):
        print(freq.shape)
        for channel in freq.T:
            padded_channel = np.pad(np.abs(channel).reshape(1, -1), (0, 24410-len(channel)), mode='constant') 
            X_test = np.pad(channel, (0, 24410-len(channel)), mode='constant')
            print(X_test.shape)
            X_test = X_test.reshape(1, -1)
            X_test = sklearn.preprocessing.minmax_scale(X_test)
            print(self.chord_clf.predict(X_test), self.chord_clf.predict_proba(X_test))
            # print(padded_channel.shape)
            # print(self.chord_clf.predict(padded_channel))


    def plot_scores(self):
        pd.DataFrame(self.chord_scores).plot()
        plt.show()

    def detect_signal(self, signal):
        var = np.var(signal)
        mean = np.mean(signal)
        if np.any((signal > mean+4*var)) and np.any(signal > 0.05):
            self.sound_detected = True
            return True
        return False
    
    def reset_scores(self):
        print(self.counter.most_common())
        #self.sound_detected = False
        self.counter.clear()

def rec_reference_sound(duration: float, samplerate: int = 44100, channels: int = 2):
    recording = sd.rec(duration*samplerate)
    sd.wait()
    return recording

def save_reference_sound(filename:str, recording: np.ndarray, samplerate: int = 44100, channels: int = 2):
    sf.write(filename, recording, samplerate=samplerate)

def detect_signal(signal_data: np.ndarray):
    '''Returns true if a signal is detected, such that the peak signal intencity is located in the first quarter of the signal data'''
    detected = False
    if np.max(signal_data) >= 0.05:
        if np.argmax(signal_data) <= len(signal_data)//5:
            detected = True
    return detected

def classify_spectrum(signal: np.ndarray, cls):
    print(signal.shape)
    for channel in signal.T:
        X_test = channel[:1024] #np.pad(channel, (0, 24410-len(channel)), mode='constant')
        X_test = X_test.reshape(1, -1)
        X_tr = sklearn.preprocessing.minmax_scale(X_test)
        print(cls.predict(X_tr), cls.predict_proba(X_tr))
    pd.DataFrame(signal).plot()
    plt.show()

def pipe_spectrum(signal: np.ndarray, pipe):
    for channel in signal.T:
        sample = channel.reshape(1,-1)
        sample = chord_classifier.apply_window(pd.DataFrame(sample))
        print(pipe.predict(sample), pipe.predict_proba(sample))

def chroma_classify(signal: np.ndarray, chroma_clf: sklearn.ensemble.RandomForestClassifier):
    for channel in signal.T:
        sample = pd.DataFrame(chroma.sample_chroma(pd.Series(channel))).T
        print(sample)
        print(chroma_clf.predict(sample), chroma_clf.predict_proba(sample))

def stream_animation():
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    # a_fft = audio_fft.AudioFFT()
    # a_fft.set_file(os.path.abspath(r'2025-2_3-audio\chords\c-chord.wav'))
    # data = a_fft.get_data()
    # a_fft.set_fdata().set_frequencies()
    #signal_data = np.zeros(data.shape)
    #freq_data = np.zeros((data.shape[0]//2, data.shape[1]))
    #ref_sounds = ReferenceSounds(os.path.abspath(r'2025-2_3-audio\chords')).cut_to_size()
    #cls = chord_classifier.get_chord_classifier()
    # pipe = chord_classifier.get_pipe_selected()
    # window_size = pipe.steps[0][1].n_features_in_*2
    #window_size, chroma_clf = chroma.get_chroma_clf()
    window_size, chroma_clf = chroma.load_model()
    print(window_size)
    signal_data = np.zeros((window_size,2))
    freq_data = np.zeros((signal_data.shape[0]//2, signal_data.shape[1]))

    def ani_update(frame, signal_data, freq_data):
        stream_data = []
        while True:
            try:
                stream_data = q.get_nowait()
            except queue.Empty:
                break
            for i, channel_data in enumerate(stream_data.T):
                signal_data.T[i] = np.roll(signal_data.T[i], -len(channel_data))
                freq_data.T[i] = np.abs(np.fft.rfft(signal_data.T[i], norm='backward')[:-1])
                #signal_data.T[i][-len(channel_data):] = channel_data ## Also works
            signal_data[-len(stream_data):, :] = stream_data

        for channel, line in enumerate(signal_lines):
            line.set_ydata(signal_data[:, channel])
        for channel, line in enumerate(freq_lines):
            line.set_ydata(freq_data[:, channel])

        # if np.any(stream_data) and ref_sounds.detect_signal(signal_data):
        #     #print('ok :-)', frame)
        #     #print(ref_sounds.classify_freq(freq_data))
        #     #ref_sounds.classify_freq(freq_data)
        #     ref_sounds.classify_freq2(freq_data)
        # elif ref_sounds.sound_detected:
        #     ref_sounds.reset_scores()
        if detect_signal(signal_data):
            print(f'signal detected {frame}')
            #classify_spectrum(freq_data, cls)
            #pipe_spectrum(freq_data, pipe)
            chroma_classify(signal_data, chroma_clf)
        return (*signal_lines, *freq_lines) # must return a single sequence of artists

    fig, axs = plt.subplots(2,1, figsize=(20,8))
    signal_lines = axs[0].plot(signal_data)
    freq_lines = axs[1].plot(freq_data)
    axs[0].legend(signal_lines, list(range(len(signal_data[0]))), title='Channel')
    axs[1].legend(signal_lines, list(range(len(freq_data[0]))), title='Channel')

    axs[0].set_ylim((-0.2, 0.2))
    axs[1].set_ylim((-.1, 150))
    q = queue.Queue()
    stream = sd.InputStream(callback=audio_callback)
    animation = FuncAnimation(fig, 
                              functools.partial(ani_update, 
                                                signal_data=signal_data, 
                                                freq_data=freq_data), 
                              interval=10, blit=True)
    with stream:
        #animation.save('stream_example.gif', writer='Pillow')
        plt.show()

    # fig, axs = plt.subplots(3,1, figsize=(20,8))
    # a_fft.fdata = a_fft.fdata[:-1]
    # a_fft.set_frequencies()
    # a_fft.frequenies = a_fft.frequenies[:-1]
    # for index, chord in enumerate(ref_sounds.reference_frequencies.keys()):
    #     a_fft.frequency_plot(ref_sounds.reference_frequencies[chord], axs[index])
    # plt.show()
    #ref_sounds.plot_scores()

if __name__ == '__main__':
    sns.set_style("whitegrid")
    sd.default.samplerate = 44100
    sd.default.channels = 2
    #recording = rec_reference_sound(2)
    #save_reference_sound(os.path.abspath(f'2025-2_3-audio\\assets\\test_file2.wav'), recording)
    stream_animation()
    #ReferenceSounds(os.path.abspath(r'2025-2_3-audio\chords'))