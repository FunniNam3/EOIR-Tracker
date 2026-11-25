"""
Center-cursor object tracker for Raspberry Pi + Arducam UC-439 (IMX219, CSI)
with HC-SR04 ultrasonic distance shown in the center of the frame.

Point the camera so the object sits under the crosshair in the middle of the
frame, press SPACE to "select" it, and a box will follow it. The tracked
object's location is printed to the terminal (and shown on screen). The
distance from the ultrasonic sensor is drawn just below the crosshair.

Controls (with a display attached):
    SPACE / s : select the object under the center cursor
    r         : release the current object and go back to selecting
    q / ESC   : quit

Wiring (BCM numbering, defaults): TRIG = GPIO 6, ECHO = GPIO 5.
    IMPORTANT: the HC-SR04 ECHO pin outputs 5 V. Use a voltage divider
    (e.g. 1k + 2k) before the Pi's 3.3 V GPIO input.

Usage:
    python3 object_tracker_distance.py
    python3 object_tracker_distance.py --trig 6 --echo 5
    python3 object_tracker_distance.py --no-sensor
    python3 object_tracker_distance.py --headless --auto-select 5
"""

import argparse
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np


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
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Center-cursor object tracker (Raspberry Pi / UC-439)")
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
    args = parser.parse_args()

    if args.headless and args.auto_select <= 0:
        print("Note: --headless has no keyboard input; setting --auto-select 5", file=sys.stderr)
        args.auto_select = 5.0

    # Ultrasonic sensor (optional: the tracker still works if it can't be opened)
    sensor = None
    if not args.no_sensor:
        try:
            sensor = UltrasonicSensor(args.trig, args.echo)
        except Exception as e:
            print(
                f"Warning: ultrasonic sensor unavailable ({e}); continuing without it.\n"
                "On Raspberry Pi 5 install: sudo apt install python3-rpi-lgpio",
                file=sys.stderr,
            )

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

    try:
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
                    label = f"center=({cx}, {cy}) box=({x}, {y}, {bw}, {bh})"
                    if frame_idx % args.print_every == 0:
                        dist_str = f" distance={distance:.1f} cm" if distance is not None else ""
                        print(f"frame {frame_idx}: {label}{dist_str}")
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

            if display is not None:
                draw_cursor(display)
                if sensor is not None:
                    draw_center_distance(display, distance)
                put_text(display, f"{fps_val:.1f} FPS", (10, display.shape[0] - 10), (255, 255, 0), 0.5)
                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord(" "), ord("s")) and not tracking:
                    tracker = select_object(frame)
                    tracking = True
                elif key == ord("r"):
                    tracker, tracking = None, False
                    print("Tracking reset")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        src.release()
        if sensor is not None:
            sensor.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()