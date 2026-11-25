import RPi.GPIO as GPIO
import time

# --- Pins (BCM numbering) ---
EN_PIN = 25
STEP_PIN = 24
DIR_PIN = 23

# --- Motor / driver settings: CHANGE THESE TO MATCH YOUR HARDWARE ---
# BTT TMC2209 V1.3 in STEP/DIR mode + 17HS8401 NEMA17 (1.8 deg = 200 steps/rev)
FULL_STEPS_PER_REV = 200   # 1.8 deg motor = 200, 0.9 deg motor = 400
# TMC2209 microsteps are set by the MS1/MS2 pins (not by this script):
#   MS2 MS1 -> 0 0 = 1/8 (default if left unconnected), 0 1 = 1/32,
#              1 0 = 1/64, 1 1 = 1/16
# The chip interpolates to 256 microsteps internally, so smoothness is the same.
MICROSTEPS = 8
GEAR_RATIO = 96 / 24       # 24T motor gear driving a 96T ring = 4:1
                           # (motor revolutions per 1 revolution of the yaw ring)
ENABLE_ACTIVE_LOW = True   # TMC2209: EN low = motor enabled
CW_LEVEL = GPIO.HIGH       # DIR level that gives clockwise; swap if it spins backwards

# --- Motion settings ---
TEST_ANGLE = 90.0          # degrees
START_DELAY = 0.002        # seconds between step edges at start (slow)
MIN_DELAY = 0.0005         # seconds between step edges at cruise speed (fast)
RAMP_STEPS = 100           # steps used to accelerate/decelerate
PAUSE = 1.0                # seconds to pause between moves

STEPS_PER_DEG = FULL_STEPS_PER_REV * MICROSTEPS * GEAR_RATIO / 360.0


def enable(on):
    level = GPIO.LOW if (on == ENABLE_ACTIVE_LOW) else GPIO.HIGH
    GPIO.output(EN_PIN, level)


def move_degrees(angle, clockwise=True):
    """Rotate the yaw axis by `angle` degrees with a simple linear speed ramp."""
    steps = int(round(angle * STEPS_PER_DEG))
    ramp = min(RAMP_STEPS, steps // 2)

    GPIO.output(DIR_PIN, CW_LEVEL if clockwise else (not CW_LEVEL))
    time.sleep(0.001)  # direction setup time

    for i in range(steps):
        # Ramp up at the start, down at the end, cruise in between.
        if i < ramp:
            frac = i / ramp
        elif i >= steps - ramp:
            frac = (steps - 1 - i) / ramp
        else:
            frac = 1.0
        delay = START_DELAY - (START_DELAY - MIN_DELAY) * frac

        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(delay)


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(EN_PIN, GPIO.OUT)
GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
enable(False)

print(f"Steps per degree: {STEPS_PER_DEG:.3f}  "
      f"({int(round(TEST_ANGLE * STEPS_PER_DEG))} steps for {TEST_ANGLE:.0f} deg)")
print("Yaw test: 90 deg CW, back, 90 deg CCW, back. Ctrl+C to stop.\n")

try:
    enable(True)
    time.sleep(0.2)

    print("Spinning 90 deg clockwise...")
    move_degrees(TEST_ANGLE, clockwise=True)
    time.sleep(PAUSE)

    print("Returning to start (90 deg counter-clockwise)...")
    move_degrees(TEST_ANGLE, clockwise=False)
    time.sleep(PAUSE)

    print("Spinning 90 deg counter-clockwise...")
    move_degrees(TEST_ANGLE, clockwise=False)
    time.sleep(PAUSE)

    print("Returning to start (90 deg clockwise)...")
    move_degrees(TEST_ANGLE, clockwise=True)

    print("Done.")
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    enable(False)  # release the motor
    GPIO.cleanup()