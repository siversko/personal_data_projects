import os
import functools

import pygame
import numpy
import scipy
import matplotlib.pyplot as plt
import scipy.integrate


class Chord(pygame.sprite.Sprite):
    simulation_time = 0
    def __init__(self, chord_name, target_surface: pygame.surface.Surface, pos: tuple[int,int] = (0,0), tempo_ms=1000, num_chords=1, *groups):
        super().__init__(*groups)
        img = os.path.join(os.path.dirname(__file__), 'img', chord_name + '.png')
        self.image = pygame.image.load(img)
        self.rect = self.image.get_rect()
        self.rect.midleft = pos # (pos[0]+self.rect.width//2, pos[1])
        self.x_pos = float(self.rect.x)

        '''Speed variables'''
        self.tempo_ms = tempo_ms
        self.t_half_beat = tempo_ms//2
        self.t_v_max     = tempo_ms//8
        print(target_surface.get_width())
        self.v_max       = 2/1.1 * (target_surface.get_width()) / (num_chords+1) / (self.t_half_beat - self.t_v_max) ## 1.1 as v_min == 0.1*v_max   2/5 *
        self.v_min       = 0.1 * self.v_max


    def update(self, time, dt, *args, **kwargs):
        
        self.x_pos -= self.get_speed(time) * dt
        self.rect.x = int(self.x_pos)
        #self.rect.x -= self.get_speed(time) * dt
        if self.rect.x < -1.5*self.rect.width:
            self.kill()

        if time%300 < 10:
            print(self.rect, self.get_speed(time))
            
        return super().update(*args, **kwargs)


        #print(time, self.tempo_ms, rel_time)
            #print('rel_time <= self.t_v_max')
            #print('rel_time >= self.t_half_beat')
    def get_speed(self, time):
        rel_time = time % self.tempo_ms
        if rel_time <= self.t_v_max:
            return self.v_max * rel_time/self.t_v_max 
        if rel_time >= self.t_half_beat:
            return 0
        return ((self.t_half_beat - rel_time)/(self.t_half_beat - self.t_v_max))*(self.v_max - self.v_min)+self.v_min

    def _plot_speed_profile(self):
        ''' Plotting speed profile for testing purpose'''
        times = numpy.linspace(0, self.tempo_ms, 100000)
        speeds = []
        for time in times:
            speeds.append(self.get_speed(time))
        speeds = numpy.array(speeds)
        fig, ax = plt.subplots()
        ax.plot(times, speeds)
        plt.show()
        print(scipy.integrate.quad(self.get_speed, 0, self.tempo_ms))
        print(scipy.integrate.quad(self.get_speed, self.tempo_ms*5, self.tempo_ms*6))


if __name__ == "__main__":
    test_surface = pygame.surface.Surface((1000,500))
    a = Chord('A-chord', test_surface)
    c = Chord('C-chord', test_surface)
    d = Chord('D-chord', test_surface)
    e = Chord('E-chord', test_surface, num_chords=2)
    print(a.image)
    print(c.image)
    print(d.image)
    print(e.image)
    e._plot_speed_profile()