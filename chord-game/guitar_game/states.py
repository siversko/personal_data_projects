import pygame

class StateManager():
    def __init__(self, surface: pygame.surface.Surface):
        self.surface = surface
        self.state_stack = []

    def load_state(self):
        pass