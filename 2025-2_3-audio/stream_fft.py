import os, queue
import audio_fft
import functools
import numpy as np
import seaborn as sns
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def rec_reference_sound(duration: float, samplerate: int = 44100, channels: int = 2):
    recording = sd.rec(duration*samplerate)
    sd.wait()
    return recording

def save_reference_sound(filename:str, recording: np.ndarray, samplerate: int = 44100, channels: int = 2):
    sf.write(filename, recording, samplerate=samplerate)


def stream_animation():
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    a_fft = audio_fft.AudioFFT()
    a_fft.set_file(os.path.abspath(r'2025-2_3-audio\chords\c-chord.wav'))
    data = a_fft.get_data()
    signal_data = np.zeros(data.shape)
    freq_data = np.zeros((data.shape[0]//2, data.shape[1]))
    

    def ani_update(frame, signal_data, freq_data):
        #print(frame, q.qsize())
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
            #print(channel, line)
            line.set_ydata(freq_data[:, channel])
        #print(signal_data)
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
        animation.save('stream_example.gif', writer='Pillow')
        plt.show()

if __name__ == '__main__':
    sd.default.samplerate = 44100
    sd.default.channels = 2
    #recording = rec_reference_sound(2)
    #save_reference_sound(os.path.abspath(f'2025-2_3-audio\\assets\\test_file2.wav'), recording)
    stream_animation()