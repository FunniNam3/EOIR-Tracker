import RPi.GPIO as GPIO
import time

SWITCH_PIN = 16  # BCM GPIO 16 (Physical Pin 36)

# --- Configure to match your wiring --------------------------------------
NORMALLY_CLOSED = True   # True = wired to NC, False = wired to NO
WIRED_TO = "GND"         # other switch terminal goes to "3V3" or "GND"
# -------------------------------------------------------------------------

# Use the internal resistor that holds the pin at the opposite level
# of whatever the switch connects it to.
pull = GPIO.PUD_DOWN if WIRED_TO == "3V3" else GPIO.PUD_UP

# Level the pin reads when the switch is connected through to the pin.
connected_level = GPIO.HIGH if WIRED_TO == "3V3" else GPIO.LOW

# NO: connected when pressed.  NC: connected when idle (open when pressed).
pressed_level = (not connected_level) if NORMALLY_CLOSED else connected_level
pressed_level = GPIO.HIGH if pressed_level else GPIO.LOW

GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=pull)

print(f"Limit switch on GPIO {SWITCH_PIN} "
      f"({'NC' if NORMALLY_CLOSED else 'NO'}, wired to {WIRED_TO}).")
print("Click the limit switch by hand. Press Ctrl+C to stop.\n")
try:
    while True:
        state = GPIO.input(SWITCH_PIN)
        status = "PRESSED (Triggered)" if state == pressed_level else "OPEN (Idle)"
        print(f"GPIO {SWITCH_PIN} State: {state} -> {status}      ", end="\r")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    GPIO.cleanup()