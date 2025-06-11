import os

import joblib
import librosa
import pandas as pd
import numpy as np
import sklearn.ensemble

def load_chroma():
    dirpath  = os.path.dirname(__file__)
    print(dirpath)
    return joblib.load(os.path.join(dirpath, 'chroma.joblib'))

def get_notes() -> list[str]:
    '''Returns a list of notes in the chromatic scale'''
    return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def sample_chroma(sample: np.ndarray, sr: int =44100):
    '''Calculates the mean chroma value accross an entire time series'''
    #print(sample)
    chromas = librosa.feature.chroma_stft(y=sample, sr=sr)
    return pd.Series(np.mean(chromas, axis = 1), index = get_notes())

def chroma_classify(signal: np.ndarray, clf: sklearn.ensemble.RandomForestClassifier):
    '''Returns predicted label and confidence probability score'''
    #print(librosa.to_mono(signal.T))
    sample = pd.DataFrame(sample_chroma(librosa.to_mono(signal.T)))
    #print(clf.predict(sample.T), np.max(clf.predict_proba(sample.T,)))
    return clf.predict(sample.T), np.max(clf.predict_proba(sample.T,))



if __name__ == '__main__':
    win_size, clf = load_chroma()
    win = np.random.random_sample((win_size,2))*(5+5) - 5
    print(chroma_classify(win, clf))