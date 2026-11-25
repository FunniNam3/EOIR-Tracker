import RPi.GPIO as GPIO
import time
import threading
import pygame

class TMC2209StepperAxis:
    def __init__(self, step_pin: int, dir_pin: int, enable_pin: int = None,
                 limit_switch_pin: int = None, step_angle: float = 1.8, 
                 microstepping: int = 8, gear_ratio: float = 1.0,
                 min_angle: float = None, max_angle: float = None, active_low: bool = True):
        
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.enable_pin = enable_pin
        self.limit_switch_pin = limit_switch_pin
        self.active_low = active_low

        self.steps_per_rev = (360.0 / step_angle) * microstepping * gear_ratio
        self.steps_per_degree = self.steps_per_rev / 360.0

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.use_soft_limits = (min_angle is not None) and (max_angle is not None)

        self.current_angle = 0.0
        self.current_step = 0

        GPIO.setup(self.step_pin, GPIO.OUT)
        GPIO.setup(self.dir_pin, GPIO.OUT)
        GPIO.output(self.step_pin, GPIO.LOW)
        GPIO.output(self.dir_pin, GPIO.LOW)

        if self.enable_pin is not None:
            GPIO.setup(self.enable_pin, GPIO.OUT)
            self.enable()

        if self.limit_switch_pin is not None:
            pud_setting = GPIO.PUD_UP if self.active_low else GPIO.PUD_DOWN
            GPIO.setup(self.limit_switch_pin, GPIO.IN, pull_up_down=pud_setting)

    def is_switch_triggered(self) -> bool:
        if self.limit_switch_pin is None:
            return False
        state = GPIO.input(self.limit_switch_pin)
        return state == GPIO.LOW if self.active_low else state == GPIO.HIGH

    def enable(self):
        if self.enable_pin is not None:
            GPIO.output(self.enable_pin, GPIO.LOW)

    def disable(self):
        if self.enable_pin is not None:
            GPIO.output(self.enable_pin, GPIO.HIGH)

    def step_relative(self, delta_angle: float, speed_deg_per_sec: float = 30.0) -> bool:
        target_angle = self.current_angle + delta_angle

        if self.use_soft_limits:
            if target_angle < self.min_angle or target_angle > self.max_angle:
                return False

        if delta_angle < 0 and self.is_switch_triggered():
            return False

        steps_to_move = int(round(abs(delta_angle) * self.steps_per_degree))
        if steps_to_move == 0:
            return True

        direction = GPIO.HIGH if delta_angle > 0 else GPIO.LOW
        GPIO.output(self.dir_pin, direction)

        steps_per_sec = max(1.0, speed_deg_per_sec * self.steps_per_degree)
        pulse_delay = max(0.00005, 1.0 / (2.0 * steps_per_sec))
        step_increment = 1 if delta_angle > 0 else -1

        for _ in range(steps_to_move):
            if delta_angle < 0 and self.is_switch_triggered():
                print(f"\n[HARD STOP] Tilt Switch Triggered at {self.current_angle:.2f}°")
                if self.use_soft_limits:
                    self.current_angle = self.min_angle
                    self.current_step = int(round(self.min_angle * self.steps_per_degree))
                return False

            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(pulse_delay)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(pulse_delay)

            self.current_step += step_increment
            self.current_angle = self.current_step / self.steps_per_degree

        return True

    def home(self, speed_deg_per_sec: float = 15.0) -> bool:
        if self.limit_switch_pin is None:
            print("[HOME WARN] No hardware limit switch assigned to this axis.")
            return False

        print("Homing Tilt axis downward (press button to simulate switch contact)...")
        GPIO.output(self.dir_pin, GPIO.LOW)
        steps_per_sec = max(1.0, speed_deg_per_sec * self.steps_per_degree)
        pulse_delay = max(0.00005, 1.0 / (2.0 * steps_per_sec))

        max_steps = int(360.0 * self.steps_per_degree)
        steps_taken = 0

        while not self.is_switch_triggered():
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(pulse_delay)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(pulse_delay)
            steps_taken += 1
            if steps_taken > max_steps:
                print("[HOME FAIL] Switch not detected within safe step limit.")
                return False

        # Back off switch slightly (+3 degrees upward)
        GPIO.output(self.dir_pin, GPIO.HIGH)
        backoff_steps = int(3.0 * self.steps_per_degree)
        for _ in range(backoff_steps):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(pulse_delay)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(pulse_delay)

        self.current_angle = (self.min_angle + 3.0) if self.use_soft_limits else 0.0
        self.current_step = int(round(self.current_angle * self.steps_per_degree))
        print(f"[HOME SUCCESS] Calibrated Tilt position to {self.current_angle:.2f}°")
        return True


class PanTiltController:
    def __init__(self, pan_pins: dict, tilt_pins: dict, tilt_limits=(-45.0, 45.0), microstepping=8):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.pan = TMC2209StepperAxis(
            step_pin=pan_pins['step'],
            dir_pin=pan_pins['dir'],
            enable_pin=pan_pins.get('enable'),
            limit_switch_pin=None,
            min_angle=None,
            max_angle=None,
            microstepping=microstepping
        )

        self.tilt = TMC2209StepperAxis(
            step_pin=tilt_pins['step'],
            dir_pin=tilt_pins['dir'],
            enable_pin=tilt_pins.get('enable'),
            limit_switch_pin=tilt_pins.get('limit_switch'),
            min_angle=tilt_limits[0],
            max_angle=tilt_limits[1],
            microstepping=microstepping
        )

        self.lock = threading.Lock()

    def home_tilt(self):
        with self.lock:
            self.tilt.home()

    def move_relative(self, pan_delta: float, tilt_delta: float, pan_speed: float, tilt_speed: float):
        with self.lock:
            if abs(pan_delta) >= 0.05:
                self.pan.step_relative(pan_delta, speed_deg_per_sec=pan_speed)
            if abs(tilt_delta) >= 0.05:
                self.tilt.step_relative(tilt_delta, speed_deg_per_sec=tilt_speed)

    def get_telemetry(self) -> dict:
        with self.lock:
            return {
                "pan_angle": round(self.pan.current_angle % 360.0, 2),
                "pan_total_degrees": round(self.pan.current_angle, 2),
                "tilt_angle": round(self.tilt.current_angle, 2),
                "tilt_switch_triggered": self.tilt.is_switch_triggered()
            }

    def cleanup(self):
        self.pan.disable()
        self.tilt.disable()
        GPIO.cleanup()


class JoystickController:
    def __init__(self, pan_tilt_system: PanTiltController, deadzone: float = 0.15, max_speed: float = 40.0):
        self.pt = pan_tilt_system
        self.deadzone = deadzone
        self.max_speed = max_speed

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No USB Joystick/Gamepad detected. Connect controller.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Joystick connected: {self.joystick.get_name()}")

    def process_input_loop(self, poll_interval: float = 0.05):
        print("Joystick tracking active. Left Stick Pan/Tilt. Ctrl+C to stop.")
        try:
            while True:
                pygame.event.pump()

                raw_x = self.joystick.get_axis(0)
                raw_y = self.joystick.get_axis(1)

                norm_x = raw_x if abs(raw_x) > self.deadzone else 0.0
                norm_y = -raw_y if abs(raw_y) > self.deadzone else 0.0

                if norm_x != 0.0 or norm_y != 0.0:
                    pan_speed = abs(norm_x) * self.max_speed
                    tilt_speed = abs(norm_y) * self.max_speed

                    pan_delta = norm_x * self.max_speed * poll_interval
                    tilt_delta = norm_y * self.max_speed * poll_interval

                    self.pt.move_relative(pan_delta, tilt_delta, pan_speed, tilt_speed)

                    telem = self.pt.get_telemetry()
                    print(f"Pan: {telem['pan_angle']}° | Tilt: {telem['tilt_angle']}° | Switch: {telem['tilt_switch_triggered']}   ", end="\r")

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\nJoystick control loop ended.")


if __name__ == "__main__":
    PAN_PINS = {'step': 17, 'dir': 27, 'enable': 22}
    TILT_PINS = {
        'step': 23, 
        'dir': 24, 
        'enable': 25,
        'limit_switch': 12
    }

    pt_system = PanTiltController(
        pan_pins=PAN_PINS,
        tilt_pins=TILT_PINS,
        tilt_limits=(-45.0, 45.0),
        microstepping=8
    )

    try:
        user_home = input("Home Tilt axis using hardware switch? (y/n): ").strip().lower()
        if user_home == 'y':
            pt_system.home_tilt()

        js = JoystickController(pt_system, deadzone=0.15, max_speed=35.0)
        js.process_input_loop(poll_interval=0.05)
    finally:
        pt_system.cleanup()
        print("\nGPIO resources safely released.")
