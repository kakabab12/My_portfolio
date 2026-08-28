import pytest

from led_status_node.controller import LedController


def test_states_and_blinking():
    outputs = []
    controller = LedController(lambda green, red: outputs.append((green, red)))

    controller.set_state('idle')
    controller.set_state('normal')
    controller.tick()
    controller.tick()
    controller.set_state('anomaly')
    controller.tick()
    controller.set_state('off')

    assert outputs == [
        (True, True), (False, False), (True, False), (False, False),
        (False, False), (False, True), (False, False),
    ]


def test_invalid_state():
    controller = LedController(lambda _green, _red: None)
    with pytest.raises(ValueError):
        controller.set_state('unknown')
