import os
import random

import pygame
import numpy
import scipy
import matplotlib.pyplot as plt
import scipy.integrate


class Chord(pygame.sprite.Sprite):

    rng = numpy.random.default_rng(seed=0)
    chord_types = numpy.array([chord_label.strip('.png') for chord_label in os.listdir(os.path.join(os.path.dirname(__file__), 'img'))], dtype=object)

    simulation_time = 0
    next_beat_time = 0
    num_active_chords = 0
    def __init__(self, chord_name, target_surface: pygame.surface.Surface, pos: tuple[int,int] = (0,0), *groups):
        super().__init__(*groups)
        print(chord_name, 'im alive!', Chord.num_active_chords ,end=' ')
        self.chord_name = chord_name
        img = os.path.join(os.path.dirname(__file__), 'img', chord_name + '.png')
        self.image = pygame.image.load(img)
        self.rect = self.image.get_rect()
        self.rect.topleft = pos # (pos[0]+self.rect.width//2, pos[1])
        self.x_pos = float(self.rect.x)
        self.target_surface = target_surface
        Chord.num_active_chords +=1

    @classmethod
    def create_Chord(cls, target_surface: pygame.surface.Surface, pos: tuple[int,int] = (0,0), *groups):
        return Chord(cls.rng.choice(cls.chord_types), target_surface, pos, *groups)

    @classmethod
    def cls_set_speed_rectangular(cls, tempo_ms:float =1000, num_chords:int=1):
        pass

    def set_speed_rectangular(self, tempo_ms:float =1000, num_chords:int=1):
        '''Speed variables for rectangular speed profile '''
        self.speed_profile = 'rectangular'
        self.tempo_ms = tempo_ms
        self.t_half_beat = tempo_ms//2
        
        self.x_n         = self.target_surface.get_width() / num_chords
        self.v_max       = 2 * self.x_n / self.t_half_beat

        self.get_speed = self.get_speed_rectangular
        return self
    
    def get_speed_rectangular(self, rel_time):
        if rel_time <= 3*self.tempo_ms//4 and rel_time >= self.tempo_ms//4:
            return self.x_n/self.t_half_beat
        return 0
    
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
    
    
    def set_speed_constant(self, tempo_ms:float = 1000, num_chords:int =1):
        self.tempo_ms = tempo_ms
        Chord.movement_settings['x_n'] = self.target_surface.get_width() / num_chords
        Chord.movement_settings['v'] = Chord.movement_settings['x_n'] / tempo_ms
        self.get_speed = self.get_speed_constant
        return self

    def get_speed_constant(self, rel_time):
        return Chord.movement_settings['x_n']*rel_time

        

    def update(self, fixed_dt, *args, **kwargs):
        #print(self.x_pos, self.get_speed(self.simulation_time), fixed_dt)
        #print(self.chord_name, self.simulation_time, self.get_speed(self.simulation_time % self.tempo_ms) * fixed_dt)
        self.x_pos -= self.get_speed(self.simulation_time % self.tempo_ms) * fixed_dt
        #self.x_pos -= self.get_speed(self.next_beat_time - self.simulation_time) * fixed_dt
        self.rect.x = int(self.x_pos)
        if self.rect.x < -1.5*self.rect.width:
            self.kill()
            Chord.num_active_chords -= 1
        return super().update(*args, **kwargs)


        #print(time, self.tempo_ms, rel_time)
            #print('rel_time <= self.t_v_max')
            #print('rel_time >= self.t_half_beat')
    def get_speed_triangular(self, rel_time):
        if rel_time <= self.t_v_max:
            return self.v_max * rel_time/self.t_v_max 
        if rel_time >= self.t_half_beat:
            return 0
        return ((self.t_half_beat - rel_time)/(self.t_half_beat - self.t_v_max))*(self.v_max - self.v_min)+self.v_min


    def _plot_speed_profile(self):
        ''' Plotting speed profile for testing purpose'''
        times = numpy.linspace(0, 10*self.tempo_ms, 100000)
        speeds = []
        for time in times:
            speeds.append(self.get_speed(time % self.tempo_ms))
        speeds = numpy.array(speeds)
        fig, ax = plt.subplots()
        ax.plot(times, speeds)
        plt.show()
        print(scipy.integrate.quad(self.get_speed, 0, self.tempo_ms, limit=1000))
        print(scipy.integrate.quad(self.get_speed, self.tempo_ms*5, self.tempo_ms*6, limit=1000))


if __name__ == "__main__":
    test_surface = pygame.surface.Surface((1000,500))
    Chord.create_Chord(test_surface)
    a = Chord('A-chord', test_surface)
    c = Chord('C-chord', test_surface)
    d = Chord('D-chord', test_surface)
    e = Chord('E-chord', test_surface).set_speed_rectangular(num_chords=4)
    print(a.image)
    print(c.image)
    print(d.image)
    print(e.image)
    e._plot_speed_profile()