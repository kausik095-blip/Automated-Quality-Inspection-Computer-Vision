"""
DecodeLabs Project 2: Automated Quality Inspection (Computer Vision)
GUI Version — Live Webcam Gear Defect Detection

IPO Pipeline:
  INPUT    -> Grayscale, Gaussian Blur, Threshold
  PROCESS  -> Contours, Convex Hull, Convexity Defects
  OUTPUT   -> PASS/FAIL, Bounding Boxes, PLC Signal
"""

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox


# =============================================================================
# CONFIGURATION & DATA CLASSES
# =============================================================================

@dataclass
class InspectionConfig:
    blur_kernel: Tuple[int, int] = (5, 5)
    threshold_value: int = 127
    max_threshold: int = 255
    threshold_max: float = 25.0
    min_contour_area: int = 1000
    consecutive_fail_frames: int = 3
    pass_color: Tuple[int, int, int] = (0, 255, 0)
    fail_color: Tuple[int, int, int] = (0, 0, 255)
    hull_color: Tuple[int, int, int] = (255, 0, 0)
    contour_color: Tuple[int, int, int] = (255, 255, 0)
    capture_width: int = 640
    capture_height: int = 480
    target_fps: int = 30
    display_max_width: int = 900
    display_max_height: int = 580


@dataclass
class DefectInfo:
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    farthest_point: Tuple[int, int]
    depth: float
    bounding_box: Tuple[int, int, int, int]


@dataclass
class InspectionResult:
    status: str
    defects: List[DefectInfo] = field(default_factory=list)
    contour: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    plc_signal: int = 0


@dataclass
class FrameUpdate:
    rgb_display: np.ndarray
    annotated_bgr: np.ndarray
    result: InspectionResult
    frame_count: int
    defect_counter: int
    verified_status: str


# =============================================================================
# INSPECTION ENGINE (Core CV Logic)
# =============================================================================

def preprocess_image(img: np.ndarray, config: InspectionConfig) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.blur_kernel, 0)
    _, thresh = cv2.threshold(
        blurred, config.threshold_value, config.max_threshold, cv2.THRESH_BINARY
    )
    return thresh


def find_main_contour(thresh: np.ndarray, config: InspectionConfig) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < config.min_contour_area:
        return None
    return contour


def compute_convexity_defects(contour: np.ndarray) -> Tuple[Optional[np.ndarray], List[DefectInfo]]:
    if len(contour) < 3:
        return None, []

    hull = cv2.convexHull(contour, returnPoints=False)
    raw_defects = cv2.convexityDefects(contour, hull)
    if raw_defects is None:
        return hull, []

    defects = []
    for i in range(raw_defects.shape[0]):
        s, e, f, d_raw = raw_defects[i, 0]
        depth = d_raw / 256.0
        start_pt = tuple(contour[s][0])
        end_pt = tuple(contour[e][0])
        farthest_pt = tuple(contour[f][0])
        pts = np.array([start_pt, end_pt, farthest_pt])
        x, y, w, h = cv2.boundingRect(pts)
        defects.append(DefectInfo(start_pt, end_pt, farthest_pt, depth, (x, y, w, h)))

    return hull, defects


def evaluate_defects(defects: List[DefectInfo], config: InspectionConfig) -> Tuple[str, List[DefectInfo]]:
    critical = [d for d in defects if d.depth > config.threshold_max]
    return ("FAIL", critical) if critical else ("PASS", [])


def draw_overlay(img, contour, hull_pts, result, config):
    output = img.copy()
    cv2.drawContours(output, [contour], -1, config.contour_color, 2)
    cv2.drawContours(output, [hull_pts], -1, config.hull_color, 2)

    for d in result.defects:
        x, y, w, h = d.bounding_box
        cv2.rectangle(output, (x, y), (x + w, y + h), config.fail_color, 2)
        cv2.circle(output, d.farthest_point, 5, (0, 255, 255), -1)
        cv2.putText(
            output, f"depth={d.depth:.1f}px", (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.fail_color, 1
        )

    color = config.fail_color if result.status == "FAIL" else config.pass_color
    label = "[ FAIL: STRUCTURAL DEFECT ]" if result.status == "FAIL" else "[ PASS: PART OK ]"
    cv2.putText(output, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return output


def draw_no_part(img):
    output = img.copy()
    cv2.putText(
        output, "[ NO PART DETECTED ]", (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
    )
    return output


def inspect_frame(frame, config):
    start = time.perf_counter()
    thresh = preprocess_image(frame, config)
    contour = find_main_contour(thresh, config)

    if contour is None:
        elapsed = (time.perf_counter() - start) * 1000
        return (
            InspectionResult("NO_PART", processing_time_ms=elapsed),
            draw_no_part(frame),
        )

    _, all_defects = compute_convexity_defects(contour)
    status, critical = evaluate_defects(all_defects, config)
    elapsed = (time.perf_counter() - start) * 1000

    result = InspectionResult(
        status=status,
        defects=critical,
        contour=contour,
        processing_time_ms=elapsed,
        plc_signal=1 if status == "FAIL" else 0,
    )
    hull_pts = cv2.convexHull(contour, returnPoints=True)
    annotated = draw_overlay(frame, contour, hull_pts, result, config)
    return result, annotated


def prepare_display_rgb(frame_bgr: np.ndarray, config: InspectionConfig) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(
        config.display_max_width / w,
        config.display_max_height / h,
        1.0,
    )
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rgb


class TripleVerificationGate:
    def __init__(self, required=3):
        self.required = required
        self.fail_streak = 0
        self.confirmed = False

    def update(self, status):
        if status == "FAIL":
            self.fail_streak += 1
        else:
            self.fail_streak = 0
            self.confirmed = False

        if self.fail_streak >= self.required and not self.confirmed:
            self.confirmed = True
            return "FAIL", True

        if self.fail_streak >= self.required:
            return "FAIL", False

        return "PASS", False

    def reset(self):
        self.fail_streak = 0
        self.confirmed = False


# =============================================================================
# GUI APPLICATION
# =============================================================================

class GearInspectionGUI:
    BG_DARK = "#0d1117"
    BG_PANEL = "#161b22"
    BG_CARD = "#1c2333"
    ACCENT_BLUE = "#00b4d8"
    ACCENT_GREEN = "#00c853"
    ACCENT_RED = "#ff1744"
    ACCENT_YELLOW = "#ffd600"
    TEXT_PRIMARY = "#e6edf3"
    TEXT_SECONDARY = "#8b949e"
    BORDER = "#30363d"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DecodeLabs — Automated Quality Inspection")
        self.root.configure(bg=self.BG_DARK)
        self.root.geometry("1280x720")
        self.root.minsize(1100, 650)

        self.config = InspectionConfig()
        self.gate = TripleVerificationGate(self.config.consecutive_fail_frames)

        self.cap = None
        self.running = False
        self.frame_queue: queue.Queue[FrameUpdate] = queue.Queue(maxsize=1)
        self.capture_thread = None

        self.frame_count = 0
        self.defect_counter = 0
        self.last_result = InspectionResult("IDLE")
        self.verified_status = "IDLE"
        self.current_frame = None
        self.photo = None

        self._build_ui()
        self._update_display()

    def _build_ui(self):
        header = tk.Frame(self.root, bg=self.BG_PANEL, height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="DECODELABS", font=("Consolas", 11, "bold"),
            fg=self.ACCENT_BLUE, bg=self.BG_PANEL
        ).pack(side=tk.LEFT, padx=(20, 0), pady=15)

        tk.Label(
            header, text="Automated Quality Inspection System",
            font=("Segoe UI", 16, "bold"), fg=self.TEXT_PRIMARY, bg=self.BG_PANEL
        ).pack(side=tk.LEFT, padx=15, pady=15)

        self.status_dot = tk.Label(
            header, text="● OFFLINE", font=("Consolas", 11),
            fg=self.ACCENT_RED, bg=self.BG_PANEL
        )
        self.status_dot.pack(side=tk.RIGHT, padx=20, pady=15)

        body = tk.Frame(self.root, bg=self.BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_frame = tk.Frame(
            body, bg=self.BG_CARD,
            highlightbackground=self.BORDER, highlightthickness=1
        )
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            video_frame, text="LIVE FEED", font=("Consolas", 10, "bold"),
            fg=self.TEXT_SECONDARY, bg=self.BG_CARD
        ).pack(anchor=tk.W, padx=10, pady=(8, 0))

        self.video_label = tk.Label(
            video_frame, bg="#000000",
            text="Camera not started\n\nClick 'START CAMERA' to begin",
            fg=self.TEXT_SECONDARY, font=("Segoe UI", 12)
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        panel = tk.Frame(
            body, bg=self.BG_PANEL, width=320,
            highlightbackground=self.BORDER, highlightthickness=1
        )
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        self._build_verdict_panel(panel)
        self._build_stats_panel(panel)
        self._build_controls_panel(panel)
        self._build_settings_panel(panel)
        self._build_action_buttons(panel)

        footer = tk.Frame(self.root, bg=self.BG_PANEL, height=30)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        self.footer_label = tk.Label(
            footer, text="Ready — Press START CAMERA to begin",
            font=("Consolas", 9), fg=self.TEXT_SECONDARY,
            bg=self.BG_PANEL, anchor=tk.W
        )
        self.footer_label.pack(fill=tk.X, padx=15, pady=5)

    def _build_verdict_panel(self, parent):
        frame = tk.Frame(parent, bg=self.BG_CARD, highlightbackground=self.BORDER, highlightthickness=1)
        frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(
            frame, text="INSPECTION VERDICT", font=("Consolas", 9, "bold"),
            fg=self.TEXT_SECONDARY, bg=self.BG_CARD
        ).pack(pady=(8, 0))

        self.verdict_label = tk.Label(
            frame, text="IDLE", font=("Segoe UI", 28, "bold"),
            fg=self.TEXT_SECONDARY, bg=self.BG_CARD, width=10, height=1
        )
        self.verdict_label.pack(pady=10)

        self.plc_label = tk.Label(
            frame, text="PLC Signal: —", font=("Consolas", 10),
            fg=self.TEXT_SECONDARY, bg=self.BG_CARD
        )
        self.plc_label.pack(pady=(0, 10))

    def _build_stats_panel(self, parent):
        frame = tk.Frame(parent, bg=self.BG_CARD, highlightbackground=self.BORDER, highlightthickness=1)
        frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            frame, text="SYSTEM METRICS", font=("Consolas", 9, "bold"),
            fg=self.TEXT_SECONDARY, bg=self.BG_CARD
        ).pack(pady=(8, 5))

        metrics = [
            ("Frames Processed", "frames_var"),
            ("Confirmed Defects", "defects_var"),
            ("Processing Time", "time_var"),
            ("Defects in Frame", "frame_defects_var"),
            ("Verified Status", "verified_var"),
        ]
        self.metrics = {}
        for label_text, var_name in metrics:
            row = tk.Frame(frame, bg=self.BG_CARD)
            row.pack(fill=tk.X, padx=15, pady=2)
            tk.Label(
                row, text=label_text, font=("Segoe UI", 9),
                fg=self.TEXT_SECONDARY, bg=self.BG_CARD, width=18, anchor=tk.W
            ).pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            tk.Label(
                row, textvariable=var, font=("Consolas", 10, "bold"),
                fg=self.TEXT_PRIMARY, bg=self.BG_CARD, anchor=tk.E
            ).pack(side=tk.RIGHT)
            self.metrics[var_name] = var

        tk.Frame(frame, height=8, bg=self.BG_CARD).pack()

    def _build_controls_panel(self, parent):
        frame = tk.Frame(parent, bg=self.BG_CARD, highlightbackground=self.BORDER, highlightthickness=1)
        frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            frame, text="CAMERA CONTROLS", font=("Consolas", 9, "bold"),
            fg=self.TEXT_SECONDARY, bg=self.BG_CARD
        ).pack(pady=(8, 5))

        cam_row = tk.Frame(frame, bg=self.BG_CARD)
        cam_row.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(
            cam_row, text="Camera Index:", font=("Segoe UI", 9),
            fg=self.TEXT_SECONDARY, bg=self.BG_CARD
        ).pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        ttk.Combobox(
            cam_row, textvariable=self.camera_var, values=["0", "1", "2"],
            width=5, state="readonly"
        ).pack(side=tk.RIGHT)

        tk.Frame(frame, height=8, bg=self.BG_CARD).pack()

    def _build_settings_panel(self, parent):
        frame = tk.Frame(parent, bg=self.BG_CARD, highlightbackground=self.BORDER, highlightthickness=1)
        frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            frame, text="TUNING PARAMETERS", font=("Consolas", 9, "bold"),
            fg=self.TEXT_SECONDARY, bg=self.BG_CARD
        ).pack(pady=(8, 5))

        sliders = [
            ("Defect Threshold (px)", "threshold_max", 5, 60, 25),
            ("Binary Threshold", "threshold_value", 50, 200, 127),
            ("Min Contour Area", "min_contour_area", 200, 5000, 1000),
            ("Blur Kernel Size", "blur_size", 1, 15, 5),
        ]
        self.slider_vars = {}
        for label, attr, lo, hi, default in sliders:
            row = tk.Frame(frame, bg=self.BG_CARD)
            row.pack(fill=tk.X, padx=15, pady=3)
            tk.Label(
                row, text=label, font=("Segoe UI", 8),
                fg=self.TEXT_SECONDARY, bg=self.BG_CARD, anchor=tk.W
            ).pack(fill=tk.X)
            var = tk.DoubleVar(value=default)
            val_label = tk.Label(
                row, text=str(int(default)), font=("Consolas", 9),
                fg=self.ACCENT_BLUE, bg=self.BG_CARD
            )
            val_label.pack(anchor=tk.E)

            def make_callback(a=attr, vl=val_label):
                def on_change(val):
                    vl.config(text=f"{int(float(val))}")
                    if a == "threshold_max":
                        self.config.threshold_max = float(val)
                    elif a == "threshold_value":
                        self.config.threshold_value = int(float(val))
                    elif a == "min_contour_area":
                        self.config.min_contour_area = int(float(val))
                    elif a == "blur_size":
                        k = int(float(val))
                        if k % 2 == 0:
                            k += 1
                        self.config.blur_kernel = (k, k)
                return on_change

            scale = tk.Scale(
                row, from_=lo, to=hi, orient=tk.HORIZONTAL,
                variable=var, showvalue=False, bg=self.BG_CARD,
                fg=self.TEXT_PRIMARY, highlightthickness=0,
                troughcolor=self.BORDER, activebackground=self.ACCENT_BLUE,
                command=make_callback()
            )
            scale.pack(fill=tk.X)
            self.slider_vars[attr] = var

        tk.Frame(frame, height=8, bg=self.BG_CARD).pack()

    def _build_action_buttons(self, parent):
        frame = tk.Frame(parent, bg=self.BG_PANEL)
        frame.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)

        btn_style = {
            "font": ("Segoe UI", 10, "bold"), "width": 28, "height": 1,
            "relief": tk.FLAT, "cursor": "hand2"
        }

        self.start_btn = tk.Button(
            frame, text="▶  START CAMERA", bg=self.ACCENT_GREEN,
            fg="#000000", activebackground="#00a844",
            command=self.start_camera, **btn_style
        )
        self.start_btn.pack(pady=3)

        self.stop_btn = tk.Button(
            frame, text="■  STOP CAMERA", bg=self.ACCENT_RED,
            fg="#ffffff", activebackground="#d50000",
            command=self.stop_camera, state=tk.DISABLED, **btn_style
        )
        self.stop_btn.pack(pady=3)

        tk.Button(
            frame, text="📷  SAVE SCREENSHOT", bg=self.BG_CARD,
            fg=self.TEXT_PRIMARY, activebackground=self.BORDER,
            command=self.save_screenshot, **btn_style
        ).pack(pady=3)

        tk.Button(
            frame, text="↻  RESET COUNTERS", bg=self.BG_CARD,
            fg=self.TEXT_PRIMARY, activebackground=self.BORDER,
            command=self.reset_counters, **btn_style
        ).pack(pady=3)

    def start_camera(self):
        cam_idx = int(self.camera_var.get())

        # V4L2 backend is faster/more reliable on Linux
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_idx)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                f"Cannot open camera {cam_idx}.\n"
                "Try a different index or close other apps using the webcam."
            )
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.running = True
        self.gate.reset()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_dot.config(text="● ONLINE", fg=self.ACCENT_GREEN)
        self.footer_label.config(text=f"Camera {cam_idx} active — Inspection running")

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop_camera(self):
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_dot.config(text="● OFFLINE", fg=self.ACCENT_RED)
        self.verdict_label.config(text="IDLE", fg=self.TEXT_SECONDARY)
        self.footer_label.config(text="Camera stopped")

    def _capture_loop(self):
        frame_interval = 1.0 / max(1, self.config.target_fps)

        while self.running and self.cap is not None:
            loop_start = time.perf_counter()

            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, self._on_capture_error)
                break

            self.frame_count += 1
            result, annotated = inspect_frame(frame, self.config)

            if result.status == "FAIL":
                verified, count_now = self.gate.update("FAIL")
                if count_now:
                    self.defect_counter += 1
            elif result.status == "PASS":
                verified, _ = self.gate.update("PASS")
            else:
                verified = "NO_PART"
                self.gate.update("PASS")

            packet = FrameUpdate(
                rgb_display=prepare_display_rgb(annotated, self.config),
                annotated_bgr=annotated,
                result=result,
                frame_count=self.frame_count,
                defect_counter=self.defect_counter,
                verified_status=verified,
            )

            try:
                self.frame_queue.put_nowait(packet)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(packet)

            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.running = False

    def _on_capture_error(self):
        self.stop_camera()
        messagebox.showerror("Camera Error", "Camera stopped responding. Stream ended.")

    def _update_display(self):
        try:
            packet = self.frame_queue.get_nowait()
            self.last_result = packet.result
            self.verified_status = packet.verified_status
            self.frame_count = packet.frame_count
            self.defect_counter = packet.defect_counter
            self.current_frame = packet.annotated_bgr

            img = Image.fromarray(packet.rgb_display)
            self.photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=self.photo, text="")
        except queue.Empty:
            pass

        self._update_metrics()
        self.root.after(33, self._update_display)

    def _update_metrics(self):
        r = self.last_result
        status = r.status

        if status == "PASS":
            self.verdict_label.config(text="PASS", fg=self.ACCENT_GREEN)
            self.plc_label.config(text="PLC Signal: PASS (0)", fg=self.ACCENT_GREEN)
        elif status == "FAIL":
            self.verdict_label.config(text="FAIL", fg=self.ACCENT_RED)
            self.plc_label.config(text="PLC Signal: FAIL (1)", fg=self.ACCENT_RED)
        elif status == "NO_PART":
            self.verdict_label.config(text="NO PART", fg=self.ACCENT_YELLOW)
            self.plc_label.config(text="PLC Signal: —", fg=self.TEXT_SECONDARY)
        else:
            self.verdict_label.config(text="IDLE", fg=self.TEXT_SECONDARY)
            self.plc_label.config(text="PLC Signal: —", fg=self.TEXT_SECONDARY)

        self.metrics["frames_var"].set(str(self.frame_count))
        self.metrics["defects_var"].set(str(self.defect_counter))
        self.metrics["time_var"].set(f"{r.processing_time_ms:.0f} ms")
        self.metrics["frame_defects_var"].set(str(len(r.defects)))
        self.metrics["verified_var"].set(self.verified_status)

    def save_screenshot(self):
        if self.current_frame is None:
            messagebox.showwarning("No Frame", "No frame available. Start the camera first.")
            return
        filename = f"inspection_{int(time.time())}.jpg"
        cv2.imwrite(filename, self.current_frame)
        messagebox.showinfo("Saved", f"Screenshot saved as:\n{filename}")
        self.footer_label.config(text=f"Screenshot saved: {filename}")

    def reset_counters(self):
        self.frame_count = 0
        self.defect_counter = 0
        self.gate.reset()
        self.footer_label.config(text="Counters reset")

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GearInspectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()