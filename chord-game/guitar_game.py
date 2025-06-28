from guitar_game import *

import pygame 
import numpy as np
import sounddevice
import queue

def set_game_events(total_time_ms, bpms):
    beat = pygame.event.custom_type()
    pygame.time.set_timer(beat, bpms, total_time_ms//bpms) # beat every bpms repeat total_time//bpms times
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

def main():
    ### Window and canvas sizes ###
    CANVAS_HEIGHT, CANVAS_WIDTH = 1600, 2000
    WINDOW_HEIGHT, WINDOW_WIDTH = 600, 1000

    sound_window, clf = model.load_chroma()
    signal_data = np.zeros((sound_window,2))

    q = queue.Queue()
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    pygame.init()
    display_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    display_canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    running = True 

    ## Font
    font = pygame.font.SysFont('Arial', 120)
    timer_text = font.render("15:00", True, "black")
    timer_text_rect = timer_text.get_rect(topright=(CANVAS_WIDTH,0))#CANVAS_HEIGHT))


    clock = pygame.time.Clock()
    prev_time = pygame.time.get_ticks()
    dt = 0
    fixed_dt = 1/120 * 1000 #every 1/120 ms

    num_chords=4
    settings = get_settings()
    beat, end_of_game = set_game_events(settings['total_time_ms'], bpm_to_ms(settings['bpm']))
    tempo_ms = bpm_to_ms(settings['bpm'])
    buffer = int(CANVAS_WIDTH*0.01)
    c_group = pygame.sprite.GroupSingle()
    c = notations.Chord('C-chord', target_surface=display_canvas, pos = (CANVAS_WIDTH, CANVAS_HEIGHT//2), tempo_ms=1000, num_chords=num_chords) # (CANVAS_WIDTH*(num_chords-1)/num_chords, CANVAS_HEIGHT//2)
    c_group.add(c)

    chords = pygame.sprite.Group()
    # chords.add(notations.Chord('C-chord', 
    #                            target_surface=display_canvas,
    #                            pos = (CANVAS_WIDTH, CANVAS_HEIGHT//2),
    #                            tempo_ms= tempo_ms,#1.5 * 1000, 
    #                            num_chords=num_chords)
    #            )
    active_chord_surface = pygame.surface.Surface((c.rect.width, c.rect.height),pygame.SRCALPHA, 32)
    active_chord_rect = active_chord_surface.get_rect()
    active_chord_rect.topleft = (0, CANVAS_HEIGHT//3)
    #active_chord_rect.midleft = (CANVAS_WIDTH//num_chords, CANVAS_HEIGHT//3)
    active_chord_surface.convert_alpha()
    active_chord_surface.fill((255,255,0,180))

    curr_beat_time = pygame.time.get_ticks()
    with sounddevice.InputStream(callback=audio_callback) as stream:
        while running:
            time = pygame.time.get_ticks()
            dt += time-prev_time
            prev_time = time

            w, h = pygame.display.get_surface().get_size()
            w_canvas, h_canvas = display_canvas.get_size()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == beat:
                    print('beat', time-curr_beat_time)
                    curr_beat_time = time
                    chords.add(notations.Chord('C-chord', 
                            target_surface=display_canvas,
                            pos = (w_canvas, h_canvas//2),
                            tempo_ms= tempo_ms,#1.5 * 1000, 
                            num_chords=num_chords)
                            )
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
            
            if model.detect_signal(signal_data):
                model.chroma_classify(signal_data, clf)

            notations.Chord.simulation_time += fixed_dt
            while fixed_dt <= dt:
                chords.update(fixed_dt)
                dt -= fixed_dt

            time_remaining = settings['total_time_ms'] - pygame.time.get_ticks()
            if not time_remaining <= 0:
                timer_text = font.render(f"{time_remaining//60_000:02}:{time_remaining // 1000 % 60:02}", True, "black")


            display_canvas.fill('white')
            #c_group.update(time, dt)
            #c_group.draw(display_canvas)
            chords.draw(display_canvas)
            #active_chord_surface.get_rect().midleft = (w_canvas//num_chords, h_canvas//3)
            display_canvas.blit(active_chord_surface, active_chord_rect)
            #display_canvas.blit(active_chord_surface, (-40, h_canvas//3))
            display_canvas.blit(timer_text, timer_text_rect)
            display_window.blit(pygame.transform.scale(display_canvas, (w, h)), (0,0))

            pygame.display.update()

    pygame.quit()
    #sys.exit()
if __name__ == "__main__":
    main()

    