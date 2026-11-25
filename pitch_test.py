
import RPi.GPIO as GPIO
import time
import sys

# --- Pins (BCM numbering) ---
EN_PIN = 22
STEP_PIN = 27
DIR_PIN = 17
LIMIT_PIN = 16

# --- Motor / driver settings ---
FULL_STEPS_PER_REV = 200
MICROSTEPS = 8
GEAR_RATIO = 1.0

ENABLE_ACTIVE_LOW = True

# Confirmed direction mapping:
UP_LEVEL = GPIO.LOW
DOWN_LEVEL = GPIO.HIGH

# --- Limit switch wiring ---
NORMALLY_CLOSED = True
WIRED_TO = "GND"

# --- Motion settings ---
TEST_ANGLE = 90.0
MAX_ANGLE = 180.0

HOME_DELAY = 0.002
HOME_MAX_DEG = 200.0
HOME_BACKOFF_DEG = 2.0

START_DELAY = 0.002
MIN_DELAY = 0.0006
RAMP_STEPS = 100
PAUSE = 1.0

STEPS_PER_DEG = FULL_STEPS_PER_REV * MICROSTEPS * GEAR_RATIO / 360.0
MAX_STEPS = int(round(MAX_ANGLE * STEPS_PER_DEG))

# --- Limit switch logic ---
PULL = GPIO.PUD_DOWN if WIRED_TO == "3V3" else GPIO.PUD_UP
_connected = GPIO.HIGH if WIRED_TO == "3V3" else GPIO.LOW
_opposite = GPIO.LOW if _connected == GPIO.HIGH else GPIO.HIGH
PRESSED_LEVEL = _opposite if NORMALLY_CLOSED else _connected

position_steps = 0


def enable(on):
    GPIO.output(
        EN_PIN,
        GPIO.LOW if (on == ENABLE_ACTIVE_LOW) else GPIO.HIGH
    )


def limit_pressed():
    return GPIO.input(LIMIT_PIN) == PRESSED_LEVEL


def pulse(delay):
    GPIO.output(STEP_PIN, GPIO.HIGH)
    time.sleep(delay)
    GPIO.output(STEP_PIN, GPIO.LOW)
    time.sleep(delay)


def home():
    """Lower pitch until the limit switch triggers."""
    global position_steps

    print("Homing: lowering pitch...")

    if limit_pressed():
        print("Limit switch already pressed; backing off.")
    else:
        GPIO.output(DIR_PIN, DOWN_LEVEL)
        time.sleep(0.01)

        max_steps = int(round(HOME_MAX_DEG * STEPS_PER_DEG))

        for _ in range(max_steps):
            if limit_pressed():
                break
            pulse(HOME_DELAY)
        else:
            raise RuntimeError(
                f"Limit switch not reached within {HOME_MAX_DEG:.0f} degrees."
            )

    # Raise away from the limit switch.
    GPIO.output(DIR_PIN, UP_LEVEL)
    time.sleep(0.01)

    for _ in range(int(round(HOME_BACKOFF_DEG * STEPS_PER_DEG))):
        pulse(HOME_DELAY)

    position_steps = 0
    print("Homing complete. Position = 0 degrees.")


def move_to_angle(angle):
    """Move to an absolute angle above zero."""
    global position_steps

    target = int(round(angle * STEPS_PER_DEG))
    target = max(0, min(MAX_STEPS, target))

    delta = target - position_steps

    if delta == 0:
        return

    going_up = delta > 0

    GPIO.output(DIR_PIN, UP_LEVEL if going_up else DOWN_LEVEL)
    time.sleep(0.01)

    steps = abs(delta)
    ramp = min(RAMP_STEPS, steps // 2)

    for i in range(steps):

        # Stop if lowering limit switch is triggered.
        if not going_up and limit_pressed():
            print("Lowering limit hit; stopping.")
            return

        if ramp > 0 and i < ramp:
            frac = i / ramp
        elif ramp > 0 and i >= steps - ramp:
            frac = (steps - 1 - i) / ramp
        else:
            frac = 1.0

        delay = START_DELAY - (START_DELAY - MIN_DELAY) * frac

        pulse(delay)
        position_steps += 1 if going_up else -1


# --- GPIO initialization ---
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(EN_PIN, GPIO.OUT)
GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(LIMIT_PIN, GPIO.IN, pull_up_down=PULL)

enable(False)

print(f"Steps per degree: {STEPS_PER_DEG:.3f}")
print(f"90-degree movement: {int(round(TEST_ANGLE * STEPS_PER_DEG))} steps")
print(f"Maximum travel: {MAX_STEPS} steps")
print("Pitch test: Home -> UP 90 degrees -> Return to zero.")

try:
    enable(True)
    time.sleep(0.2)

    home()
    time.sleep(PAUSE)

    print("Raising pitch to 90 degrees...")
    move_to_angle(TEST_ANGLE)
    time.sleep(PAUSE)

    print("Returning pitch to zero...")
    move_to_angle(0)

    print(f"Done. Position: {position_steps / STEPS_PER_DEG:.2f} degrees.")

except KeyboardInterrupt:
    print("\nStopped.")

except RuntimeError as e:
    print(f"Error: {e}", file=sys.stderr)

finally:
    enable(False)
    GPIO.cleanup()
