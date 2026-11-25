"""
Object tracker + yaw/pitch follow for Raspberry Pi + Arducam UC-439 (IMX219, CSI).

Aim the center crosshair at an object and press SPACE. A box follows the
object, and the yaw (left/right) and pitch (up/down) motors move the camera to
keep the object centered. The farther the object is from the center, the faster
the camera moves. The HC-SR04 distance is drawn just below the crosshair.

At startup the pitch axis homes (lowers until the limit switch triggers), then
moves to the start angle. On exit it lowers the camera back to zero before
releasing the motor, so the camera doesn't drop.

Controls (with a display attached):
    SPACE / s : select the object under the center cursor
    r         : release the object (both motors stop and hold)
    q / ESC   : quit

Pins (BCM):
    Yaw motor:   EN = 25, STEP = 24, DIR = 23   (24T -> 96T ring, 4:1)
    Pitch motor: EN = 22, STEP = 27, DIR = 17   (1:1), lowering limit switch = 16
    Ultrasonic:  TRIG = 6, ECHO = 5             (ECHO needs a 5V -> 3.3V divider)

Usage:
    python3 tracker_yaw_pitch.py
    python3 tracker_yaw_pitch.py --no-yaw --no-pitch   # camera only
    python3 tracker_yaw_pitch.py --invert-yaw          # camera turns away horizontally
    python3 tracker_yaw_pitch.py --invert-pitch        # camera tilts away vertically
    python3 tracker_yaw_pitch.py --pitch-start 60      # angle above the limit switch to start at
"""

import argparse
import math
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Hardware settings
# --------------------------------------------------------------------------
FULL_STEPS_PER_REV = 200   # 1.8 deg motors
MICROSTEPS = 8             # set by MS1/MS2 on the TMC2209 (both low = 1/8)
ENABLE_ACTIVE_LOW = True   # TMC2209: EN low = motor enabled

# Yaw
YAW_EN_PIN, YAW_STEP_PIN, YAW_DIR_PIN = 25, 24, 23
YAW_GEAR_RATIO = 96 / 24   # 24T motor gear driving a 96T ring = 4:1
YAW_POSITIVE_HIGH = False  # DIR level for clockwise (camera turns right); flipped from the original True
YAW_STEPS_PER_DEG = FULL_STEPS_PER_REV * MICROSTEPS * YAW_GEAR_RATIO / 360.0

# Pitch
PITCH_EN_PIN, PITCH_STEP_PIN, PITCH_DIR_PIN = 22, 27, 17
PITCH_LIMIT_PIN = 16       # pitch lowering limit switch
PITCH_GEAR_RATIO = 1.0
PITCH_POSITIVE_HIGH = False  # confirmed: UP_LEVEL = LOW, DOWN_LEVEL = HIGH
PITCH_STEPS_PER_DEG = FULL_STEPS_PER_REV * MICROSTEPS * PITCH_GEAR_RATIO / 360.0

# Pitch limit switch wiring
NORMALLY_CLOSED = True
WIRED_TO = "GND"
LIMIT_PULL = None          # resolved in setup_limit_switch()
LIMIT_PRESSED_LEVEL = None

# Pitch homing
HOME_DELAY = 0.002         # seconds per step edge while homing (slow, safe)
HOME_MAX_DEG = 200.0       # give up if the switch isn't hit within this travel
HOME_BACKOFF_DEG = 2.0     # raise this far after hitting the switch; that spot is zero


def setup_limit_switch(GPIO):
    """Work out the pull resistor and the 'pressed' level for the pitch limit switch."""
    global LIMIT_PULL, LIMIT_PRESSED_LEVEL
    LIMIT_PULL = GPIO.PUD_DOWN if WIRED_TO == "3V3" else GPIO.PUD_UP
    connected = GPIO.HIGH if WIRED_TO == "3V3" else GPIO.LOW
    opposite = GPIO.LOW if connected == GPIO.HIGH else GPIO.HIGH
    LIMIT_PRESSED_LEVEL = opposite if NORMALLY_CLOSED else connected
    GPIO.setup(PITCH_LIMIT_PIN, GPIO.IN, pull_up_down=LIMIT_PULL)

    def pressed():
        return GPIO.input(PITCH_LIMIT_PIN) == LIMIT_PRESSED_LEVEL
    return pressed


# --------------------------------------------------------------------------
# Stepper axis (speed-controlled in a background thread so video never blocks)
# --------------------------------------------------------------------------
class StepperAxis:
    """
    One stepper axis. The main loop calls set_rate_dps() with a signed speed in
    degrees/second (positive = the axis's positive direction); a background
    thread generates the step pulses, ramping speed with a limited acceleration
    so the motor doesn't stall. Soft limits and an optional hard stop switch
    (for the negative direction) stop motion at the ends of travel.

    Blocking helpers (home, move_to_deg) are for use before start() / after the
    thread has stopped.
    """

    MIN_RATE = 15.0  # steps/s; below this the axis is treated as stopped

    def __init__(self, name, en, step, dir_, positive_high, steps_per_deg,
                 max_rate, accel, min_deg=None, max_deg=None,
                 stop_switch=None, extra_pins=()):
        import RPi.GPIO as GPIO  # imported here so the tracker works without it

        self.GPIO = GPIO
        self.name = name
        self.en, self.step, self.dir = en, step, dir_
        self.pos_level = GPIO.HIGH if positive_high else GPIO.LOW
        self.neg_level = GPIO.LOW if positive_high else GPIO.HIGH
        self.steps_per_deg = steps_per_deg
        self.max_rate = max_rate
        self.accel = accel
        self.min_steps = None if min_deg is None else int(round(min_deg * steps_per_deg))
        self.max_steps = None if max_deg is None else int(round(max_deg * steps_per_deg))
        self.stop_switch = stop_switch
        self.extra_pins = tuple(extra_pins)

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.en, GPIO.OUT)
        GPIO.setup(self.step, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.dir, GPIO.OUT, initial=GPIO.LOW)
        self._set_enabled(True)
        time.sleep(0.2)

        self._lock = threading.Lock()
        self._target = 0.0        # commanded steps/s
        self._speed = 0.0         # current (ramped) steps/s
        self._position = 0        # steps from zero (+ = positive direction)
        self._dir_positive = None
        self._stop = threading.Event()
        self._thread = None

    # ---- basics ----
    def _set_enabled(self, on):
        level = self.GPIO.LOW if (on == ENABLE_ACTIVE_LOW) else self.GPIO.HIGH
        self.GPIO.output(self.en, level)

    def _set_dir(self, positive):
        self.GPIO.output(self.dir, self.pos_level if positive else self.neg_level)
        self._dir_positive = positive
        time.sleep(0.0001)

    def _pulse_blocking(self, delay):
        self.GPIO.output(self.step, self.GPIO.HIGH)
        time.sleep(delay)
        self.GPIO.output(self.step, self.GPIO.LOW)
        time.sleep(delay)

    def _blocked(self, positive, pos):
        if positive:
            return self.max_steps is not None and pos >= self.max_steps
        if self.min_steps is not None and pos <= self.min_steps:
            return True
        return self.stop_switch is not None and self.stop_switch()

    @property
    def position_deg(self):
        with self._lock:
            return self._position / self.steps_per_deg

    # ---- blocking helpers (thread not running) ----
    def home(self):
        """Move in the negative direction until the stop switch triggers, back off, call it zero."""
        if self.stop_switch is None:
            raise RuntimeError(f"{self.name}: no stop switch to home against")
        print(f"Homing {self.name}: lowering until the limit switch triggers...")

        if self.stop_switch():
            print("  Limit switch already pressed; backing off first.")
        else:
            self._set_dir(False)
            time.sleep(0.01)
            for _ in range(int(round(HOME_MAX_DEG * self.steps_per_deg))):
                if self.stop_switch():
                    break
                self._pulse_blocking(HOME_DELAY)
            else:
                raise RuntimeError(
                    f"{self.name}: limit switch not reached within {HOME_MAX_DEG:.0f} deg. "
                    "Check the switch wiring and the motor direction."
                )

        self._set_dir(True)
        time.sleep(0.01)
        for _ in range(int(round(HOME_BACKOFF_DEG * self.steps_per_deg))):
            self._pulse_blocking(HOME_DELAY)

        self._position = 0
        print(f"  {self.name} zeroed.")

    def move_to_deg(self, target_deg, start_delay=0.002, min_delay=0.0006, ramp_steps=100):
        """Move to an absolute angle with a speed ramp (respects soft limits and the stop switch)."""
        target = int(round(target_deg * self.steps_per_deg))
        if self.max_steps is not None:
            target = min(target, self.max_steps)
        if self.min_steps is not None:
            target = max(target, self.min_steps)
        delta = target - self._position
        if delta == 0:
            return

        positive = delta > 0
        self._set_dir(positive)
        time.sleep(0.01)

        steps = abs(delta)
        ramp = min(ramp_steps, steps // 2)
        for i in range(steps):
            if not positive and self.stop_switch is not None and self.stop_switch():
                print(f"  {self.name}: limit switch hit; stopping.")
                return
            if ramp > 0 and i < ramp:
                frac = i / ramp
            elif ramp > 0 and i >= steps - ramp:
                frac = (steps - 1 - i) / ramp
            else:
                frac = 1.0
            self._pulse_blocking(start_delay - (start_delay - min_delay) * frac)
            self._position += 1 if positive else -1

    # ---- speed control (background thread) ----
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_rate_dps(self, rate_dps):
        """Command a signed speed in degrees/second."""
        s = max(-self.max_rate, min(self.max_rate, float(rate_dps) * self.steps_per_deg))
        with self._lock:
            self._target = s

    def _run(self):
        GPIO = self.GPIO
        last = time.perf_counter()
        next_step = last
        while not self._stop.is_set():
            now = time.perf_counter()
            dt = now - last
            last = now

            with self._lock:
                target = self._target
            # Ramp the speed toward the target.
            max_dv = self.accel * dt
            diff = target - self._speed
            self._speed = target if abs(diff) <= max_dv else self._speed + math.copysign(max_dv, diff)
            speed = self._speed

            if abs(speed) < self.MIN_RATE:
                time.sleep(0.002)
                next_step = time.perf_counter()
                continue

            positive = speed > 0
            with self._lock:
                pos = self._position
            if self._blocked(positive, pos):   # end of travel or limit switch
                self._speed = 0.0
                time.sleep(0.002)
                continue

            if positive != self._dir_positive:
                self._set_dir(positive)

            period = 1.0 / abs(speed)
            if now >= next_step:
                GPIO.output(self.step, GPIO.HIGH)
                time.sleep(0.00002)
                GPIO.output(self.step, GPIO.LOW)
                with self._lock:
                    self._position += 1 if positive else -1
                next_step += period
                if next_step < now - 0.05:   # fell behind: don't burst to catch up
                    next_step = now + period
            else:
                time.sleep(min(next_step - now, 0.002))

    def release(self, park_deg=None):
        """Stop, optionally move to park_deg (e.g. lower the camera), then release the motor."""
        self.set_rate_dps(0.0)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            if park_deg is not None:
                print(f"Parking {self.name} at {park_deg:.0f} deg before releasing...")
                self.move_to_deg(park_deg)
        except Exception as e:
            print(f"Warning: could not park {self.name}: {e}", file=sys.stderr)
        finally:
            try:
                self._set_enabled(False)  # release the motor
                self.GPIO.cleanup([self.en, self.step, self.dir, *self.extra_pins])
            except Exception:
                pass


# --------------------------------------------------------------------------
# Ultrasonic sensor (runs in a background thread so it never blocks video)
# --------------------------------------------------------------------------
class UltrasonicSensor:
    """HC-SR04 distance sensor read continuously in a background thread."""

    SPEED_FACTOR = 17150.0  # cm/s: speed of sound (34300) / 2 for round trip
    TIMEOUT = 0.03          # seconds; ~5 m round trip is ~29 ms

    def __init__(self, trig, echo, interval=0.1):
        import RPi.GPIO as GPIO  # imported here so the tracker works without it

        self.GPIO = GPIO
        self.trig = trig
        self.echo = echo
        self.interval = interval

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trig, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)
        GPIO.output(self.trig, False)
        time.sleep(0.5)  # let the sensor settle

        self._readings = deque(maxlen=3)
        self._misses = 0
        self._distance = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _measure(self):
        """One measurement in cm, or None on timeout."""
        GPIO = self.GPIO
        GPIO.output(self.trig, True)
        time.sleep(0.00001)
        GPIO.output(self.trig, False)

        deadline = time.time() + self.TIMEOUT
        pulse_start = time.time()
        while GPIO.input(self.echo) == 0:
            pulse_start = time.time()
            if pulse_start > deadline:
                return None

        deadline = time.time() + self.TIMEOUT
        pulse_end = pulse_start
        while GPIO.input(self.echo) == 1:
            pulse_end = time.time()
            if pulse_end > deadline:
                return None

        return round((pulse_end - pulse_start) * self.SPEED_FACTOR, 2)

    def _run(self):
        while not self._stop.is_set():
            d = self._measure()
            with self._lock:
                if d is None:
                    self._misses += 1
                    if self._misses >= 5:  # several failures in a row: no reading
                        self._readings.clear()
                        self._distance = None
                else:
                    self._misses = 0
                    self._readings.append(d)
                    # Median of the last few readings rejects single-sample spikes.
                    self._distance = float(np.median(self._readings))
            self._stop.wait(self.interval)

    @property
    def distance(self):
        with self._lock:
            return self._distance

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.GPIO.cleanup([self.trig, self.echo])
        except Exception:
            pass


# --------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------
class PiCameraSource:
    """Arducam UC-439 (IMX219) via Picamera2/libcamera."""

    def __init__(self, width, height, fps, camera_num=0, hflip=False, vflip=False):
        try:
            from picamera2 import Picamera2
            from libcamera import Transform
        except ImportError:
            print(
                "Error: picamera2 not found. Install with:\n"
                "  sudo apt install -y python3-picamera2\n"
                "(and if using a venv, create it with --system-site-packages)",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            self.cam = Picamera2(camera_num)
        except Exception as e:
            print(
                f"Error: could not open the CSI camera ({e}).\n"
                "Check the ribbon cable (contacts facing the right way, pushed in fully),\n"
                "run 'rpicam-hello --list-cameras', and try adding 'dtoverlay=imx219' to\n"
                "/boot/firmware/config.txt, then reboot.",
                file=sys.stderr,
            )
            sys.exit(1)

        config = self.cam.create_video_configuration(
            # "RGB888" gives pixels in B,G,R byte order, which is what OpenCV expects.
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
            transform=Transform(hflip=hflip, vflip=vflip),
            buffer_count=4,
        )
        self.cam.configure(config)
        self.cam.start()
        time.sleep(1.0)  # let auto-exposure / white balance settle

    def read(self):
        frame = self.cam.capture_array("main")
        if frame is None:
            return False, None
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return True, np.ascontiguousarray(frame)

    def release(self):
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass


class CvCaptureSource:
    """USB webcam or video file through OpenCV (handy for testing)."""

    def __init__(self, source, width, height, fps):
        if source.startswith("usb") and source[3:].isdigit():
            idx = int(source[3:])
            self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Error: could not open video source '{source}'", file=sys.stderr)
            sys.exit(1)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def open_source(args):
    if args.source == "csi":
        return PiCameraSource(
            args.width, args.height, args.fps,
            camera_num=args.camera_num, hflip=args.hflip, vflip=args.vflip,
        )
    return CvCaptureSource(args.source, args.width, args.height, args.fps)


# --------------------------------------------------------------------------
# Tracking helpers
# --------------------------------------------------------------------------
def create_tracker(name: str):
    """Create a cv2 tracker, handling the legacy namespace in contrib builds."""
    name = name.upper()
    factories = {
        "CSRT": "TrackerCSRT_create",
        "KCF": "TrackerKCF_create",
        "MIL": "TrackerMIL_create",
    }
    if name not in factories:
        raise ValueError(f"Unknown tracker '{name}'. Choose from {list(factories)}")

    fn = factories[name]
    if hasattr(cv2, fn):
        return getattr(cv2, fn)()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, fn):
        return getattr(legacy, fn)()

    raise RuntimeError(
        f"{name} tracker not available in this OpenCV build "
        f"(version {cv2.__version__}). Try --tracker KCF or MIL, or "
        "'sudo apt install python3-opencv'."
    )


def initial_box_at_center(frame, box_size=120, refine=True):
    """
    Return an (x, y, w, h) box around whatever is at the center of the frame.

    Starts with a fixed-size box centered on the cursor. If `refine` is True,
    GrabCut is used to segment the object under the cursor and tighten the box.
    Falls back to the fixed box if segmentation fails.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = box_size // 2

    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(cx + half, w - 1)
    y1 = min(cy + half, h - 1)
    default_box = (x0, y0, x1 - x0, y1 - y0)

    if not refine:
        return default_box

    try:
        pad = box_size // 2
        wx0, wy0 = max(x0 - pad, 0), max(y0 - pad, 0)
        wx1, wy1 = min(x1 + pad, w), min(y1 + pad, h)
        window = frame[wy0:wy1, wx0:wx1]

        rect = (x0 - wx0, y0 - wy0, x1 - x0, y1 - y0)
        mask = np.zeros(window.shape[:2], np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        # 3 iterations is plenty and keeps the one-off cost low on a Pi.
        cv2.grabCut(window, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)

        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
        if n <= 1:
            return default_box

        lx, ly = cx - wx0, cy - wy0
        label = labels[ly, lx]
        if label == 0:
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

        bx = stats[label, cv2.CC_STAT_LEFT] + wx0
        by = stats[label, cv2.CC_STAT_TOP] + wy0
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        if bw < 20 or bh < 20 or bw * bh > 0.9 * window.shape[0] * window.shape[1]:
            return default_box
        return (int(bx), int(by), int(bw), int(bh))
    except cv2.error:
        return default_box


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def draw_cursor(frame, color=(0, 255, 255), size=15, gap=5):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, 2)
    cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, 2)
    cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, 2)
    cv2.circle(frame, (cx, cy), 2, color, -1)


def put_text(frame, text, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_center_distance(frame, distance_cm, offset=40, scale=0.8):
    """Draw the ultrasonic distance centered horizontally, just below the crosshair."""
    h, w = frame.shape[:2]
    text = "-- cm" if distance_cm is None else f"{distance_cm:.1f} cm"
    color = (0, 0, 255) if distance_cm is None else (0, 255, 255)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    org = (w // 2 - tw // 2, h // 2 + offset + th // 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Axis control
# --------------------------------------------------------------------------
def axis_command(err_px, size_px, fov_deg, kp, deadband_px, invert):
    """
    Convert an offset from the frame center (pixels) into a signed rate in deg/s.
    Returns (rate_dps, error_deg). Inside the dead band the rate is zero.
    """
    err_deg_raw = err_px / size_px * fov_deg
    if abs(err_px) <= deadband_px:
        return 0.0, err_deg_raw
    err_px -= math.copysign(deadband_px, err_px)
    err_deg = err_px / size_px * fov_deg
    rate = kp * err_deg
    return (-rate if invert else rate), err_deg_raw


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Object tracker + yaw/pitch follow (Raspberry Pi / UC-439)")
    parser.add_argument("--source", default="csi",
                        help="'csi' (UC-439 on the ribbon port, default), 'usb0' for a USB "
                             "webcam, or a video file path")
    parser.add_argument("--camera-num", type=int, default=0,
                        help="CSI camera index if more than one is attached")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hflip", action="store_true", help="mirror image horizontally")
    parser.add_argument("--vflip", action="store_true", help="flip image vertically")
    parser.add_argument("--tracker", default="KCF",
                        help="KCF (fast, default on Pi), CSRT (more accurate, slower), MIL")
    parser.add_argument("--box-size", type=int, default=100,
                        help="size of the initial selection box around the cursor")
    parser.add_argument("--no-refine", action="store_true",
                        help="skip GrabCut refinement; use a fixed box at the cursor")
    parser.add_argument("--headless", action="store_true",
                        help="no window (for SSH / no monitor); use with --auto-select")
    parser.add_argument("--auto-select", type=float, default=0.0, metavar="SECONDS",
                        help="automatically select the object under the cursor after N seconds")
    parser.add_argument("--print-every", type=int, default=1,
                        help="print the position every N frames (default 1)")
    parser.add_argument("--trig", type=int, default=6, help="ultrasonic TRIG pin (BCM)")
    parser.add_argument("--echo", type=int, default=5, help="ultrasonic ECHO pin (BCM)")
    parser.add_argument("--no-sensor", action="store_true",
                        help="run without the ultrasonic sensor")
    parser.add_argument("--deadband", type=float, default=20.0,
                        help="pixels from center within which the camera doesn't move (default 20)")

    yaw = parser.add_argument_group("yaw control (left/right)")
    yaw.add_argument("--no-yaw", action="store_true", help="don't move the yaw motor")
    yaw.add_argument("--invert-yaw", action="store_true",
                     help="reverse yaw direction (use if the camera turns away from the object)")
    yaw.add_argument("--hfov", type=float, default=62.2,
                     help="camera horizontal field of view in degrees (default 62.2)")
    yaw.add_argument("--kp", type=float, default=1.5,
                     help="yaw gain: deg/s of turn rate per degree of error (default 1.5)")
    yaw.add_argument("--max-rate", type=float, default=1500.0,
                     help="maximum yaw step rate in steps/s (default 1500)")
    yaw.add_argument("--accel", type=float, default=4000.0,
                     help="yaw acceleration limit in steps/s^2 (default 4000)")
    yaw.add_argument("--yaw-limit", type=float, default=180.0,
                     help="stop turning beyond this many degrees either side of the start "
                          "(0 = no limit, default 180)")

    pitch = parser.add_argument_group("pitch control (up/down)")
    pitch.add_argument("--no-pitch", action="store_true", help="don't move the pitch motor")
    pitch.add_argument("--invert-pitch", action="store_true",
                       help="reverse pitch direction (use if the camera tilts away from the object)")
    pitch.add_argument("--vfov", type=float, default=48.8,
                       help="camera vertical field of view in degrees (default 48.8)")
    pitch.add_argument("--pitch-kp", type=float, default=1.5,
                       help="pitch gain: deg/s per degree of error (default 1.5)")
    pitch.add_argument("--pitch-max-rate", type=float, default=500.0,
                       help="maximum pitch step rate in steps/s (default 500)")
    pitch.add_argument("--pitch-accel", type=float, default=1500.0,
                       help="pitch acceleration limit in steps/s^2 (default 1500)")
    pitch.add_argument("--pitch-start", type=float, default=90.0,
                       help="angle above the limit switch to move to after homing (default 90)")
    pitch.add_argument("--pitch-max", type=float, default=180.0,
                       help="maximum pitch angle above zero, in degrees (default 180)")
    args = parser.parse_args()

    if args.headless and args.auto_select <= 0:
        print("Note: --headless has no keyboard input; setting --auto-select 5", file=sys.stderr)
        args.auto_select = 5.0

    sensor = None
    yaw_motor = None
    pitch_motor = None
    src = None

    try:
        # Ultrasonic sensor (optional)
        if not args.no_sensor:
            try:
                sensor = UltrasonicSensor(args.trig, args.echo)
            except Exception as e:
                print(
                    f"Warning: ultrasonic sensor unavailable ({e}); continuing without it.\n"
                    "On Raspberry Pi 5 install: sudo apt install python3-rpi-lgpio",
                    file=sys.stderr,
                )

        # Yaw motor (optional)
        if not args.no_yaw:
            try:
                limit = args.yaw_limit if args.yaw_limit > 0 else None
                yaw_motor = StepperAxis(
                    "yaw", YAW_EN_PIN, YAW_STEP_PIN, YAW_DIR_PIN, YAW_POSITIVE_HIGH,
                    YAW_STEPS_PER_DEG, args.max_rate, args.accel,
                    min_deg=-limit if limit else None, max_deg=limit,
                )
                yaw_motor.start()
            except Exception as e:
                print(f"Warning: yaw motor unavailable ({e}); continuing without it.\n"
                      "On Raspberry Pi 5 install: sudo apt install python3-rpi-lgpio",
                      file=sys.stderr)

        # Pitch motor (optional): home to the limit switch, then go to the start angle
        if not args.no_pitch:
            try:
                import RPi.GPIO as GPIO
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                pitch_motor = StepperAxis(
                    "pitch", PITCH_EN_PIN, PITCH_STEP_PIN, PITCH_DIR_PIN, PITCH_POSITIVE_HIGH,
                    PITCH_STEPS_PER_DEG, args.pitch_max_rate, args.pitch_accel,
                    min_deg=0.0, max_deg=args.pitch_max,
                    stop_switch=setup_limit_switch(GPIO), extra_pins=(PITCH_LIMIT_PIN,),
                )
                pitch_motor.home()
                time.sleep(0.5)
                print(f"Moving pitch to {args.pitch_start:.0f} deg...")
                pitch_motor.move_to_deg(args.pitch_start)
                pitch_motor.start()
            except Exception as e:
                print(f"Warning: pitch disabled ({e}).", file=sys.stderr)
                if pitch_motor is not None:
                    pitch_motor.release()   # no parking: position isn't trustworthy
                pitch_motor = None

        src = open_source(args)

        tracker = None
        tracking = False
        frame_idx = 0
        window = "Object Tracker"
        start_time = time.time()
        fps_t0, fps_frames, fps_val = time.time(), 0, 0.0

        if args.headless:
            print(f"Headless mode: will select the object under the cursor in {args.auto_select:.0f}s. Ctrl+C to quit.")
        else:
            print("SPACE/s = select object under cursor | r = reset | q/ESC = quit")

        def select_object(frame):
            box = initial_box_at_center(frame, args.box_size, refine=not args.no_refine)
            trk = create_tracker(args.tracker)
            trk.init(frame, box)
            print(f"Selected object, initial box (x, y, w, h) = {box}")
            return trk

        while True:
            ok, frame = src.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            fps_frames += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps_val = fps_frames / (now - fps_t0)
                fps_t0, fps_frames = now, 0

            distance = sensor.distance if sensor is not None else None
            frame_h, frame_w = frame.shape[:2]
            yaw_rate = 0.0      # deg/s for this frame; 0 unless we have a target
            pitch_rate = 0.0
            yaw_err = pitch_err = None

            # Auto-select (useful when headless)
            if not tracking and args.auto_select > 0 and now - start_time >= args.auto_select:
                tracker = select_object(frame)
                tracking = True

            display = frame.copy() if not args.headless else None

            if tracking:
                found, box = tracker.update(frame)
                if found:
                    x, y, bw, bh = [int(v) for v in box]
                    cx, cy = x + bw // 2, y + bh // 2
                    # Object right of center -> turn right. Object above center -> tilt up.
                    yaw_rate, yaw_err = axis_command(
                        cx - frame_w / 2.0, frame_w, args.hfov, args.kp,
                        args.deadband, args.invert_yaw)
                    pitch_rate, pitch_err = axis_command(
                        frame_h / 2.0 - cy, frame_h, args.vfov, args.pitch_kp,
                        args.deadband, args.invert_pitch)
                    label = f"center=({cx}, {cy}) box=({x}, {y}, {bw}, {bh})"
                    if frame_idx % args.print_every == 0:
                        dist_str = f" distance={distance:.1f} cm" if distance is not None else ""
                        yaw_str = f" yaw_err={yaw_err:+.1f}" if yaw_motor is not None else ""
                        pitch_str = f" pitch_err={pitch_err:+.1f}" if pitch_motor is not None else ""
                        print(f"frame {frame_idx}: {label}{dist_str}{yaw_str}{pitch_str}")
                    if display is not None:
                        cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                        cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)
                        put_text(display, "TRACKING", (10, 25), (0, 255, 0))
                        put_text(display, label, (10, 50))
                else:
                    if display is not None:
                        put_text(display, "TARGET LOST - press 'r' to reselect", (10, 25), (0, 0, 255))
                    print(f"frame {frame_idx}: target lost")
                    if args.headless:
                        # Nothing to press; go back to selecting automatically.
                        tracker, tracking = None, False
                        start_time = time.time()
            elif display is not None:
                put_text(display, "Aim cursor at object and press SPACE to select", (10, 25))

            # Send this frame's commands to the motors (0 stops them; they keep holding).
            if yaw_motor is not None:
                yaw_motor.set_rate_dps(yaw_rate)
            if pitch_motor is not None:
                pitch_motor.set_rate_dps(pitch_rate)

            if display is not None:
                draw_cursor(display)
                if sensor is not None:
                    draw_center_distance(display, distance)
                line = frame_h - 30
                if pitch_motor is not None:
                    put_text(display, f"PITCH {pitch_motor.position_deg:+.1f} deg",
                             (10, line), (255, 200, 0), 0.5)
                    line -= 20
                if yaw_motor is not None:
                    put_text(display, f"YAW {yaw_motor.position_deg:+.1f} deg",
                             (10, line), (255, 200, 0), 0.5)
                put_text(display, f"{fps_val:.1f} FPS", (10, frame_h - 10), (255, 255, 0), 0.5)
                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord(" "), ord("s")) and not tracking:
                    tracker = select_object(frame)
                    tracking = True
                elif key == ord("r"):
                    tracker, tracking = None, False
                    if yaw_motor is not None:
                        yaw_motor.set_rate_dps(0.0)
                    if pitch_motor is not None:
                        pitch_motor.set_rate_dps(0.0)
                    print("Tracking reset")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if yaw_motor is not None:
            yaw_motor.release()
        if pitch_motor is not None:
            pitch_motor.release(park_deg=0.0)   # lower the camera before letting go
        if src is not None:
            src.release()
        if sensor is not None:
            sensor.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()