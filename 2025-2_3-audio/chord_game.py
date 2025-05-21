import chroma
from stream_fft import detect_signal, chroma_classify

import pygame 
import numpy as np
import sounddevice
import librosa
import pandas as pd

import os, sys
import queue

def transform_mouse_pos(mouse_pos, display_canvas):
    pass

def main():
    CANVAS_HEIGHT, CANVAS_WIDTH = 1600, 2000
    WINDOW_HEIGHT, WINDOW_WIDTH = 600, 1000

    sound_window, clf = chroma.load_model()
    signal_data = np.zeros((sound_window,2))

    q = queue.Queue()
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    pygame.init()
    display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    display_canvas = pygame.Surface((CANVAS_HEIGHT, CANVAS_HEIGHT))
    running = True 
    with sounddevice.InputStream(callback=audio_callback) as stream:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            stream_data=[]
            while True:
                try:
                    stream_data = q.get_nowait()
                except queue.Empty:
                    break
                for i, channel_data in enumerate(stream_data.T):
                    signal_data.T[i] = np.roll(signal_data.T[i], -len(channel_data))
                signal_data[-len(stream_data):, :] = stream_data
            
            if detect_signal(signal_data):
                chroma_classify(signal_data, clf)

            display_canvas.fill('white')
            display_surface.blit(pygame.transform.scale(display_canvas, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0,0))

    pygame.quit()
    #sys.exit()
if __name__ == "__main__":
    main()

    