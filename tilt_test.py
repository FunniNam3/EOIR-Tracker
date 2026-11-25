import RPi.GPIO as GPIO
import time

STEP_PIN = 23
DIR_PIN = 24
EN_PIN = 25
SWITCH_PIN = 12

MICROSTEPPING = 8  # Change to 16 if the first motor turned 45° instead of 90°
STEPS_PER_REV = 200 * MICROSTEPPING
STEPS_PER_DEG = STEPS_PER_REV / 360.0

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([STEP_PIN, DIR_PIN, EN_PIN], GPIO.OUT)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def move_tilt(degrees, speed_deg_per_sec=25.0):
    steps = int(round(abs(degrees) * STEPS_PER_DEG))
    moving_down = degrees < 0
    GPIO.output(DIR_PIN, GPIO.LOW if moving_down else GPIO.HIGH)
    
    steps_per_sec = speed_deg_per_sec * STEPS_PER_DEG
    delay = 1.0 / (2.0 * steps_per_sec)
    
    for _ in range(steps):
        if moving_down and GPIO.input(SWITCH_PIN) == GPIO.LOW:
            print("Limit switch triggered! Stopping downward motion.")
            return
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(delay)

try:
    GPIO.output(EN_PIN, GPIO.LOW)  # Enable driver
    print("Tilting up +30 degrees...")
    move_tilt(30.0)
    time.sleep(1.0)
    
    print("Tilting down -45 degrees (press the limit switch by hand to test cutoff)...")
    move_tilt(-45.0)
    print("Tilt test complete.")
finally:
    GPIO.output(EN_PIN, GPIO.HIGH) # Disable driver
    GPIO.cleanup()
