import numpy as np

def detect_signal(signal_data: np.ndarray):
    '''Returns true if a signal is detected, such that the peak signal intencity is located in the first quarter of the signal data'''
    detected = False
    if np.max(signal_data) >= 0.05:
        if np.argmax(signal_data) <= len(signal_data)//5:
            detected = True
    return detected

if __name__ == "__main__":
    win = np.random.random_sample((2**12,2))*(5+5) - 5
    detect_signal(win)