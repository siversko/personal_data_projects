import os, json
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def _draw_frame(ax, num_strings=6):
    ax.hlines([num_strings-1], xmin=-0.01, 
                             xmax= num_strings-1+0.015, 
                             linewidth=5.0, colors=['black'])
    ax.hlines([_ for _ in range( 0,  num_strings,  1)], xmin= 0, xmax=num_strings-1, colors=['black'])
    ax.vlines([_ for _ in range( 0,  num_strings,  1)], ymin= 0, ymax=num_strings-1, colors=['black'])

def _string(ax, string, open_closed, num_strings = 6):
    if open_closed:
        open_closed = 'O'
    else:
        open_closed = 'X'
    ax.text(string-1,num_strings-0.85,str(open_closed), 
            horizontalalignment='center', 
            verticalalignment='center',
            fontdict={'size':22},
            color='black')
    return ax

def _fret(ax, fret, num_strings = 6):
    ax.text(num_strings-1+0.2, num_strings - 1.5,str(fret), 
            horizontalalignment='center', 
            verticalalignment='center',
            fontdict={'size':22},
            color='black')
    return ax


def _draw_finger(ax, string, fret, finger_num):
    ### As the frame is a 5x5 grid we transform the row to achieve the right  position
    pos = (string-1, (-fret+5.5)%5)
    print(pos)
    ax.add_artist(patches.Circle(pos, radius=0.25, color='black'))
    ax.text(pos[0],pos[1],str(finger_num), 
            horizontalalignment='center', 
            verticalalignment='center',
            fontdict={'size':22},
            color='white')
    return ax


def read_chord_diagram(chord_name):
    with open(os.path.join(os.path.dirname(__file__), 'json_diagrams', chord_name + '.json'), 'r') as f:
        chord = json.load(f)
    print(json.dumps(chord, indent=4))
    return chord

def save_chord(chord_name):
    plt.savefig(os.path.join(os.path.dirname(__file__), 'img', chord_name + '.svg'))

def chord_diagram(chord_name):
    fig, ax = plt.subplots(figsize=(4, 5.96))
    chord = read_chord_diagram(chord_name)
    num_strings = len(chord["strings"])
    _draw_frame(ax, num_strings)
    if chord['fret'] != 0:
        _fret(ax, chord['fret'], num_strings)
    for string in chord['strings']:
        if type(string['play']) == bool:
            _string(ax, string['string'], string['play'], num_strings)
            continue
        _draw_finger(ax, string['string'], string['play']['fret'], string['play']['finger'])
    plt.title(chord_name, fontdict={'size': 30})
    plt.tight_layout()
    plt.axis('off')
    save_chord(chord_name)
    plt.show()

if __name__ == '__main__':
    chord_diagram("E-chord")
    #read_chord_diagram('E-chord')