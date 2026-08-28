import pygame
import random
from constants import INTERNAL_HEIGHT

class Camera:
    def __init__(self):
        self.offset_y = 0  # Vertical offset
        self.offset_x = 0  # Horizontal offset
        self.shake_timer = 0
        self.shake_intensity = 0

    def shake(self, duration, intensity, bias_x=0, bias_y=0):
        """Start a camera shake effect with optional directional bias."""
        self.shake_timer = duration
        self.shake_intensity = intensity
        self.bias_x = bias_x
        self.bias_y = bias_y

    def update(self, target_y, smoothing=0.1):
        """Smoothly follow the target's y position (e.g., pickaxe)"""
        desired_offset = target_y - INTERNAL_HEIGHT // 2
        self.offset_y += (desired_offset - self.offset_y) * smoothing

        # Apply shake if active
        if self.shake_timer > 0:
            damping_factor = self.shake_timer / (self.shake_timer + 1)  # Gradual reduction
            self.offset_y += random.uniform(-self.shake_intensity, self.shake_intensity) * damping_factor + self.bias_y
            self.offset_x += (random.uniform(-self.shake_intensity, self.shake_intensity) * damping_factor + self.bias_x) * 0.5 # Less horizontal shake
            self.shake_timer -= 1
        else:
            self.offset_x = (0 - self.offset_x) * smoothing