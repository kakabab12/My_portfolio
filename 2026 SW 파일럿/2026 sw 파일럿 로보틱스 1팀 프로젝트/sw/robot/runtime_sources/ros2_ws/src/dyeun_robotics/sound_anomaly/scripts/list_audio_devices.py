"""Show the device number/name needed by the recording and live scripts."""

from __future__ import annotations

import sounddevice as sd


def main() -> None:
    default_input, default_output = sd.default.device
    print("Audio devices (input-capable devices are marked with [INPUT]):\n")
    for index, device in enumerate(sd.query_devices()):
        is_input = device["max_input_channels"] > 0
        marker = "[INPUT]" if is_input else "       "
        default = " (default input)" if index == default_input else ""
        print(
            f"{marker} {index:>2}: {device['name']} | "
            f"in={device['max_input_channels']}, out={device['max_output_channels']}, "
            f"default SR={device['default_samplerate']:.0f}{default}"
        )
    print(f"\nDefault output device: {default_output}")


if __name__ == "__main__":
    main()
