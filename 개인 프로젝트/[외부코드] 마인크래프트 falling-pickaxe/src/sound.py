import pygame

class SoundManager:
    def __init__(self):
        pygame.mixer.init()  # Initialize the mixer
        pygame.mixer.set_num_channels(128)
        self.sounds = {}

    def load_sound(self, name, path, volume=1.0):
        """Load a sound and set its volume"""
        sound = pygame.mixer.Sound(str(path))
        sound.set_volume(volume)
        self.sounds[name] = sound

    def play_sound(self, name, loop=False):
        """Play a loaded sound"""
        if name in self.sounds:
            self.sounds[name].play(loops=-1 if loop else 0)

    def stop_sound(self, name):
        """Stop a playing sound"""
        if name in self.sounds:
            self.sounds[name].stop()

    def stop_all(self):
        """Stop all sounds"""
        pygame.mixer.stop()
