VALID_STATES = ('idle', 'normal', 'anomaly', 'off')


class LedController:
    """State machine independent of ROS and GPIO hardware."""

    def __init__(self, write_leds):
        self._write_leds = write_leds
        self.state = 'off'
        self._phase = False

    def set_state(self, state):
        if state not in VALID_STATES:
            raise ValueError(f"invalid state '{state}'; use: {', '.join(VALID_STATES)}")
        self.state = state
        self._phase = False
        self.apply()

    def tick(self):
        if self.state in ('normal', 'anomaly'):
            self._phase = not self._phase
            self.apply()

    def apply(self):
        if self.state == 'idle':
            green, red = True, True
        elif self.state == 'normal':
            green, red = self._phase, False
        elif self.state == 'anomaly':
            green, red = False, self._phase
        else:
            green, red = False, False
        self._write_leds(green, red)

