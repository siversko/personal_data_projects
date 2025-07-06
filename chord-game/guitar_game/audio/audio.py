import os

def get_audio():
    return os.path.join(os.path.dirname(__file__), 'box_hit.wav')
    #return os.path.join(os.path.dirname(__file__), 'guitar_slap.wav')
    
def get_sample():
    return os.path.join(os.path.dirname(__file__), 'sample-a.wav')

if __name__ == '__main__':
    audio = get_audio()
    print(audio)