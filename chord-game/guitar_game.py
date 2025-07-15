from guitar_game import *

import pygame 
import numpy as np
import sounddevice
import queue

def set_game_events(total_time_ms, bpms, num_chords):
    beat = pygame.event.custom_type()
    pygame.time.set_timer(beat, bpms, total_time_ms//bpms + num_chords) # beat every bpms repeat total_time//bpms times
    end_of_game = pygame.event.custom_type()
    pygame.time.set_timer(end_of_game, total_time_ms, True) # repeat once
    return beat, end_of_game

def transform_mouse_pos(mouse_pos, display_canvas):
    pass 

def get_settings():
    settings = {"chords" : ["A", "C", "E", "D"],
                "bpm" : 30,#120, #120, #20, # 105,
                "total_time_ms" : round(0.3*60_000)}    
    return settings

def bpm_to_ms(bpm):
    # beat/min -> ms/beat = (ms/min) / (beat / min)
    # Must be an int number of ms
    return 60_000 // bpm

def populate_signal(window_array):
    ''' returns a soundfile to classify on. because first classification lags '''
    import soundfile
    sample = soundfile.read(audio.get_sample())
    return sample[0][:len(window_array)]


def get_score_field(score_text: pygame.Surface, time_offset_text: pygame.Surface, clf_pred_text: pygame.Surface, score_color: tuple[int,int,int,int], scale: float = 1) -> pygame.Surface:
    score_text_rect = score_text.get_rect()
    time_offset_rect = time_offset_text.get_rect()
    clf_pred_rect = clf_pred_text.get_rect()
    score_field = pygame.surface.Surface((1.4*(max(time_offset_rect.width + clf_pred_rect.width, score_text_rect.width)), 
                                        1.0*(score_text_rect.height + clf_pred_rect.height)),pygame.SRCALPHA, 32)
    score_field.fill(score_color)
    score_field_rect = score_field.get_rect()
    score_text_rect.midtop       = (score_field_rect.width//2, 0)
    time_offset_rect.bottomright = (score_field_rect.width, score_field_rect.height)
    clf_pred_rect.bottomleft     = (0, score_field_rect.height)

    score_field.blit(score_text, score_text_rect)
    score_field.blit(time_offset_text, time_offset_rect)
    score_field.blit(clf_pred_text, clf_pred_rect)
    score_field = pygame.transform.scale(score_field, (int(score_field_rect.width*scale), int(score_field_rect.height*scale)))
    return score_field

def main():
    ### Window and canvas sizes ###
    CANVAS_HEIGHT, CANVAS_WIDTH = 1600, 2000
    WINDOW_HEIGHT, WINDOW_WIDTH = 600, 1000

    sound_window, clf = model.load_chroma('chroma_ovr.joblib')
    signal_data = np.zeros((sound_window,2))
    _ = model.chroma_classify(populate_signal(signal_data), clf)

    q = queue.Queue()
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    pygame.init()

    num_chords=4
    settings = get_settings()
    beat, end_of_game = set_game_events(settings['total_time_ms'], bpm_to_ms(settings['bpm']), num_chords)
    num_beats = settings['total_time_ms']//bpm_to_ms(settings['bpm'])
    beat_sound = pygame.mixer.Sound(audio.get_audio())
    beat_sound.set_volume(1.5)
    tempo_ms = bpm_to_ms(settings['bpm'])

    display_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    display_canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    running = True 
    recently_detected = False
    time_delay = 0

    ## Font
    font = pygame.font.SysFont('Arial', 120)
    timer_text = font.render("15:00", True, "black")
    timer_text_rect = timer_text.get_rect(topright=(CANVAS_WIDTH,0))#CANVAS_HEIGHT))
    beats_text = font.render(f"{num_beats}", True, "black")
    beats_text_rect = beats_text.get_rect(topright=(CANVAS_WIDTH,timer_text_rect.height))
    

    ### Score Info ###
    score, beat_score = 0,0
    score_text = font.render(f"Score: {score:5}(+{beat_score:3})", True, "black")
    score_text_rect = score_text.get_rect()

    time_sign = '+'
    time_offset_text = font.render(f"{time_sign}{bpm_to_ms(settings['bpm']):4}", True, "black")
    time_offset_rect = time_offset_text.get_rect()

    clf_pred,  clf_proba =  ['--'], 0
    clf_pred_text = font.render(f"{clf_pred} {clf_proba:2}", True, "black")
    clf_pred_rect = clf_pred_text.get_rect()
    score_color = (0,0,0,180)
    score_field = pygame.surface.Surface((1.4*(max(time_offset_rect.width + clf_pred_rect.width, score_text_rect.width)), 
                                          1.0*(score_text_rect.height + clf_pred_rect.height)),pygame.SRCALPHA, 32)
    score_field.fill(score_color)
    score_field_rect = score_field.get_rect()
    score_text_rect.midtop       = (score_field_rect.width//2, 0)
    time_offset_rect.bottomright = (score_field_rect.width, score_field_rect.height)
    clf_pred_rect.bottomleft     = (0, score_field_rect.height)
    score_field_rect.midtop      = (CANVAS_WIDTH//2,0)
    prev_score_text, prev_time_offset_text, prev_clf_pred_text, prev_score_color = score_text, time_offset_text, clf_pred_text, score_color


    clock = pygame.time.Clock()
    prev_time = pygame.time.get_ticks()
    dt = 0
    fixed_dt = 1/120 * 1000 #every 1/120 ms


    buffer = int(CANVAS_WIDTH*0.01)
    c_group = pygame.sprite.GroupSingle()
    c = notations.Chord('C-chord', target_surface=display_canvas, pos = (CANVAS_WIDTH, CANVAS_HEIGHT//2)) # (CANVAS_WIDTH*(num_chords-1)/num_chords, CANVAS_HEIGHT//2)
    c_group.add(c)
    print()

    chords = pygame.sprite.Group()
    active_chord_surface = pygame.surface.Surface((c.rect.width, c.rect.height),pygame.SRCALPHA, 32)
    active_chord_rect = active_chord_surface.get_rect()
    active_chord_rect.topleft = (0, CANVAS_HEIGHT//3)
    #active_chord_rect.midleft = (CANVAS_WIDTH//num_chords, CANVAS_HEIGHT//3)
    active_chord_surface.convert_alpha()
    active_chord_surface.fill((255,255,0,180))

    curr_beat_time = pygame.time.get_ticks()
    next_beat_time = curr_beat_time + tempo_ms
    with sounddevice.InputStream(callback=audio_callback):
        while running:
            time = pygame.time.get_ticks()
            dt += time-prev_time
            prev_time = time

            w, h = pygame.display.get_surface().get_size()
            w_canvas, h_canvas = display_canvas.get_size()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == beat:
                    #print('beat', time-curr_beat_time)
                    beat_sound.play()
                    curr_beat_time = time
                    next_beat_time = time + tempo_ms
                    print(time, notations.Chord.simulation_time)
                    notations.Chord.next_beat_time = next_beat_time
                    if num_beats > 0:
                        chords.add(notations.Chord.create_Chord(target_surface=display_canvas,
                                                                pos = (w_canvas, h_canvas//3)).set_speed_rectangular(tempo_ms=tempo_ms, num_chords=num_chords))
                        num_beats -= 1
                        beats_text = font.render(f"{num_beats}", True, "black")
                    print(time)
                    if len(chords) < 1:
                        print('empty')
                if event.type == end_of_game:
                    print('the_end')
                    #running = False
            stream_data=[]
            while True:
                try:
                    stream_data = q.get_nowait()
                except queue.Empty:
                    break
                for i, channel_data in enumerate(stream_data.T):
                    signal_data.T[i] = np.roll(signal_data.T[i], -len(channel_data))
                signal_data[-len(stream_data):, :] = stream_data

            
            
            if abs(next_beat_time - time) < abs(curr_beat_time - time):
                time_sign = '-'
                time_target = abs(next_beat_time - time)
            else: 
                time_sign = '+'
                time_target = abs(curr_beat_time - time)

            if model.detect_signal(signal_data) and not recently_detected:
                
                if 'prev_score_field' in locals():
                    prev_prev_score_field = get_score_field(prev_score_text, prev_time_offset_text, prev_clf_pred_text, prev_score_color, scale=0.4)
                    prev_prev_score_rect = prev_prev_score_field.get_rect()
                    prev_prev_score_rect.topleft = (0, prev_score_rect.height)
                prev_score_field = get_score_field(score_text, time_offset_text, clf_pred_text, score_color, scale=0.5)
                prev_score_rect = prev_score_field.get_rect()
                chord_collition = active_chord_rect.collidedict({chord : chord.rect for chord in chords})
                clf_pred, clf_proba = model.chroma_classify(signal_data, clf)
                print(clf_pred, clf_proba)
                beat_score = clf_proba*max(1 - time_target/(tempo_ms//2),0)

                if chord_collition:
                    chord_collition = chord_collition[0]
                    if chord_collition.chord_name.split('-')[0].lower() != clf_pred[0]:
                        beat_score = 0
                        #print('Wrong chord!', chord_collition.chord_name.split('-')[0].lower(), clf_pred[0])


                score += beat_score*100

                recently_detected = True
                time_delay = time + int(tempo_ms*0.4)
                prev_score_text, prev_time_offset_text, prev_clf_pred_text, prev_score_color = score_text, time_offset_text, clf_pred_text, score_color
                #prev_prev_score_text, prev_prev_time_offset_text, prev_prev_clf_pred_text, prev_prev_score_color = prev_score_text, prev_time_offset_text, prev_clf_pred_text, prev_score_color

            time_offset_text = font.render(f"{time_sign}{time_target:4}", True, "black")
            clf_pred_text = font.render(f"{clf_pred[0]:<3} {clf_proba*100:.0f}", True, "black")
            score_text = font.render(f"Score: {score:.0f}(+{beat_score*100:.0f})", True, "black")
            score_color = ((255) * (1-beat_score), # r //8
                            (255) *    beat_score , # g //8
                            0,                         #b
                            180)                       #alpha
            
            if time > time_delay:
                recently_detected = False
            while fixed_dt <= dt:
                chords.update(fixed_dt)
                dt -= fixed_dt
                notations.Chord.simulation_time += fixed_dt

            time_remaining = settings['total_time_ms'] - pygame.time.get_ticks()
            if not time_remaining <= 0:
                timer_text = font.render(f"{time_remaining//60_000:02}:{time_remaining // 1000 % 60:02}", True, "black")
            if len(chords) < 1 and num_beats <= 0:
                time_offset_text = font.render(f"{time_sign}{0:4}", True, "black")


            display_canvas.fill('white')
            chords.draw(display_canvas)
            display_canvas.blit(active_chord_surface, active_chord_rect)
            display_canvas.blit(timer_text, timer_text_rect)
            display_canvas.blit(beats_text, beats_text_rect)
            if not recently_detected or not drawn_once:
                score_field.fill(score_color)#((0,0,0,180)) #score_field.fill((225,225//8,0,180))
                score_field.blit(score_text, score_text_rect)
                score_field.blit(time_offset_text, time_offset_rect)
                score_field.blit(clf_pred_text, clf_pred_rect)
                drawn_once = False
            else:
                drawn_once = True
            display_canvas.blit(score_field, score_field_rect)
            if 'prev_score_field' in locals():
                #display_canvas.blit(pygame.transform.scale(prev_score_field, (prev_score_rect.width//2, prev_score_rect.height//2)), (0,0))
                display_canvas.blit(prev_score_field, prev_score_rect)
            if 'prev_prev_score_field' in locals():
                display_canvas.blit(prev_prev_score_field, prev_prev_score_rect)
            display_window.blit(pygame.transform.scale(display_canvas, (w, h)), (0,0))

            pygame.display.update()

    pygame.quit()
    #sys.exit()
if __name__ == "__main__":
    main()
    # sound_window, clf = model.load_chroma()
    # signal_data = np.zeros((sound_window,2))
    # print(sound_window)
    # arr = populate_signal(signal_data)
    # print(model.chroma_classify(arr, clf))