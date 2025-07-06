import os
import functools

import pygame
import numpy
import scipy
import matplotlib.pyplot as plt
import scipy.integrate


class Chord(pygame.sprite.Sprite):
    simulation_time = 0
    def __init__(self, chord_name, target_surface: pygame.surface.Surface, pos: tuple[int,int] = (0,0), *groups):
        super().__init__(*groups)
        img = os.path.join(os.path.dirname(__file__), 'img', chord_name + '.png')
        self.image = pygame.image.load(img)
        self.rect = self.image.get_rect()
        self.rect.midleft = pos # (pos[0]+self.rect.width//2, pos[1])
        self.x_pos = float(self.rect.x)
        self.target_surface = target_surface


    def set_speed_rectangular(self, tempo_ms:float =1000, num_chords:int=1):
        '''Speed variables for rectangular speed profile '''
        self.speed_profile = 'rectangular'
        self.tempo_ms = tempo_ms
        self.t_half_beat = tempo_ms//2
        self.t_v_max     = tempo_ms//8
        
        self.x_n         = self.target_surface.get_width() / num_chords
        self.v_max       = 2/1.1 * self.x_n / ((self.t_half_beat - self.t_v_max)) ## 1.1 as v_min == 0.1*v_max   2/5 *
        self.v_min       = 0.1 * self.v_max

        self.get_speed = self.get_speed_rectangular
        return self
    
    def set_speed_triangular(self, tempo_ms:float =1000, num_chords:int=1):
        '''#### not correct'''
        self.tempo_ms = tempo_ms
        self.t_half_beat = tempo_ms//2
        self.t_v_max     = tempo_ms//8
        
        self.x_n = self.target_surface.get_width() / num_chords
        self.v_max       = 2/1.1 * self.x_n / ((self.t_half_beat - self.t_v_max)) ## 1.1 as v_min == 0.1*v_max   2/5 *
        self.v_min       = 0.1 * self.v_max

        self.get_speed = self.get_speed_triangular
        return self


    def update(self, fixed_dt, *args, **kwargs):
        #print(self.x_pos, self.get_speed2(self.simulation_time), fixed_dt)
        #print(self, self.simulation_time)
        self.x_pos -= self.get_speed(self.simulation_time) * fixed_dt
        self.rect.x = int(self.x_pos)
        if self.rect.x < -1.5*self.rect.width:
            self.kill()
        return super().update(*args, **kwargs)


        #print(time, self.tempo_ms, rel_time)
            #print('rel_time <= self.t_v_max')
            #print('rel_time >= self.t_half_beat')
    def get_speed_triangular(self, time):
        rel_time = time % self.tempo_ms

        if rel_time <= self.t_v_max:
            return self.v_max * rel_time/self.t_v_max 
        if rel_time >= self.t_half_beat:
            return 0
        return ((self.t_half_beat - rel_time)/(self.t_half_beat - self.t_v_max))*(self.v_max - self.v_min)+self.v_min
    
    
    def get_speed_rectangular(self, time):
        rel_time = time % self.tempo_ms
        #print(self, self.simulation_time, rel_time)
        if rel_time <= (self.tempo_ms*3)//4 and rel_time >= self.tempo_ms//4:
            return self.x_n/self.t_half_beat
        return 0


    def _plot_speed_profile(self):
        ''' Plotting speed profile for testing purpose'''
        times = numpy.linspace(0, self.tempo_ms, 100000)
        speeds = []
        for time in times:
            speeds.append(self.get_speed2(time))
        speeds = numpy.array(speeds)
        fig, ax = plt.subplots()
        ax.plot(times, speeds)
        plt.show()
        print(scipy.integrate.quad(self.get_speed2, 0, self.tempo_ms, limit=1000))
        print(scipy.integrate.quad(self.get_speed2, self.tempo_ms*5, self.tempo_ms*6, limit=1000))


if __name__ == "__main__":
    test_surface = pygame.surface.Surface((1000,500))
    a = Chord('A-chord', test_surface)
    c = Chord('C-chord', test_surface)
    d = Chord('D-chord', test_surface)
    e = Chord('E-chord', test_surface, num_chords=4)
    print(a.image)
    print(c.image)
    print(d.image)
    print(e.image)
    e._plot_speed_profile()