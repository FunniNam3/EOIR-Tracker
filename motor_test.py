import RPi.GPIO as GPIO
import time

STEP_PIN = 17
DIR_PIN = 27
EN_PIN = 22

MICROSTEPPING = 8  # Default 1/8 microstepping when MS1 & MS2 are unconnected
STEPS_PER_REV = 200 * MICROSTEPPING
STEPS_PER_DEG = STEPS_PER_REV / 360.0

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([STEP_PIN, DIR_PIN, EN_PIN], GPIO.OUT)

def move_degrees(degrees, speed_deg_per_sec=30.0):
    steps = int(round(abs(degrees) * STEPS_PER_DEG))
    GPIO.output(DIR_PIN, GPIO.HIGH if degrees > 0 else GPIO.LOW)

    steps_per_sec = speed_deg_per_sec * STEPS_PER_DEG
    delay = 1.0 / (2.0 * steps_per_sec)

    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(delay)

try:
    GPIO.output(EN_PIN, GPIO.LOW)  # Enable driver (Active LOW)
    print("Rotating +90 degrees...")
    move_degrees(90.0)
    time.sleep(1.0)

    print("Rotating -90 degrees back to start...")
    move_degrees(-90.0)
    print("Done.")
finally:
    GPIO.output(EN_PIN, GPIO.HIGH) # Disable driver outputs when finished
    GPIO.cleanup()
