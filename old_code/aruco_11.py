
# %% ===========================================================================
# IMPORTS
# =============================================================================
import cv2
import numpy as np
from cv2 import aruco
import time
import csv
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from scipy.spatial.transform import Rotation as R
import json
import shutil
from collections import deque


# --- Try to import and set a modern theme for the GUI
try:
    import sv_ttk
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False
    print("Warning: 'sv-ttk' library not found. GUI will use the default theme.")
    print("For a modern look, run: pip install sv-ttk")


# Import and configure Matplotlib for non-interactive plotting to a buffer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% ===========================================================================
# CONFIGURATION PARAMETERS
# (These are the default values. They will be overridden by 'app_settings.json' if it exists.)
# =============================================================================

# --- Camera Setup ---
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (3840, 2160)  # Natively set to 4K
CAMERA_FPS = 30
REVERSE_CAMERA_DISPLAY = True     # Default setting for reversing the live display

# --- Visualization & Overlay Scaling (Tuned for 4K) ---
SHOW_REALTIME_PLOTS = True        # Enable/disable the real-time plot panel
PLOT_WIDTH_PX = 1500              # Width of the plot panel in pixels.
INFO_PANEL_WIDTH_PX = 450         # Width of the info panel.
OVERLAY_FONT_SCALE = 1.2          # General font scale for cv2.putText.
OVERLAY_THICKNESS = 2             # General thickness for cv2.putText and lines.
AXES_THICKNESS = 6                # Thickness for cv2.drawFrameAxes.
TRAJECTORY_THICKNESS = 4          # Thickness for buoy trajectory lines.
PLOT_HISTORY_LENGTH = 150         # Number of past data points to show on plots.
PLOT_TITLE_FONTSIZE = 32
PLOT_LABEL_FONTSIZE = 32
PLOT_TICK_FONTSIZE = 32
PLOT_LINE_WIDTH = 2.0
PLOT_LEGEND_FONTSIZE = 'medium'

# --- Video Recording Settings ---
RECORD_VIDEO = False
VIDEO_CODEC = 'mp4v'
VIDEO_FPS = 30
VIDEO_EXTENSION = '.mp4'

# --- ArUco Marker Configuration ---
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
MARKER_SIZE_METERS = 0.1577

# --- Reference Marker Configuration ---
REFERENCE_MARKER_ID = 0
USE_REFERENCE_MARKER = True

# --- LED Sync Configuration ---
ENABLE_LED_DETECTION = False
LED_DETECTION_ROI = None
LED_OFF_BASELINE_BRIGHTNESS = 0.0
LED_BRIGHTNESS_THRESHOLD = 30.0

# --- Buoy Configuration ---
NUM_BUOYS = 9
MARKERS_PER_BUOY = 3
BUOY_ID_TO_MARKER_IDS_MAP = {
    i + 1: list(range(10 + i * 10, 10 + i * 10 + MARKERS_PER_BUOY))
    for i in range(NUM_BUOYS)
}
MARKER_ID_TO_BUOY_ID_MAP = {
    marker_id: buoy_id
    for buoy_id, marker_ids_list in BUOY_ID_TO_MARKER_IDS_MAP.items()
    for marker_id in marker_ids_list
}

# --- Buoy Geometry ---
TRIANGLE_SIDE_LENGTH_METERS = 0.21
_s = TRIANGLE_SIDE_LENGTH_METERS
MARKER_RELATIVE_GEOMETRY = {
    0: {'name': 'X0', 'rel_tvec_to_X0': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'rel_rvec_to_X0': np.array([0.0, 0.0, 0.0], dtype=np.float32)},
    1: {'name': 'X1', 'rel_tvec_to_X0': np.array([_s, 0.0, 0.0], dtype=np.float32),
        'rel_rvec_to_X0': np.array([0.0, np.deg2rad(120), 0.0], dtype=np.float32)},
    2: {'name': 'X2', 'rel_tvec_to_X0': np.array([_s * np.cos(np.pi/3), 0.0, _s * np.sin(np.pi/3)], dtype=np.float32),
        'rel_rvec_to_X0': np.array([0.0, np.deg2rad(240), 0.0], dtype=np.float32)}
}
BUOY_COG_OFFSET_METERS = 0.23168
BUOY_COG_OFFSET_VECTOR_ASSEMBLY_FRAME = np.array(
    [0, -BUOY_COG_OFFSET_METERS, -1/3 * TRIANGLE_SIDE_LENGTH_METERS], dtype=np.float32
)

# --- Data Output & File Structure ---
DATA_OUTPUT_DIRECTORY = "buoy_tracking_data"
EXPERIMENTS_BASE_DIRECTORY_NAME = "experiments"
CONFIG_DIRECTORY_NAME = "config"
PATTERNS_DIRECTORY_NAME = "generated_patterns"

CALIBRATION_FILE = "camera_calibration.npz"
BUOY_GEOMETRY_CALIBRATION_FILE = "buoy_marker_calibrations.json"
SETTINGS_FILE = "app_settings.json"
APP_ICON_PATH = 'aruco_icon.png'

DEFAULT_CAMERA_MATRIX = np.array([
    [1000, 0, CAMERA_RESOLUTION[0]/2],
    [0, 1000, CAMERA_RESOLUTION[1]/2],
    [0, 0, 1]], dtype=np.float32)
DEFAULT_DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

CALIBRATION_CHESSBOARD_SHAPE = (9, 6)
CHESSBOARD_SQUARE_SIZE_MM = 26.9
BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR = 5

# --- Logging & Trajectory ---
LOG_ALL_MARKER_DATA = False
TRAJECTORY_LENGTH = 30

# --- Color definitions ---
TRAJECTORY_COLOR = (0, 0, 255)
MARKER_INFO_COLOR = (0, 255, 0)
BUOY_COG_INFO_COLOR = (255, 100, 0)
TIMESTAMP_COLOR = (0, 0, 255)
REFERENCE_MARKER_COLOR = (255, 0, 255)
RECORD_INDICATOR_COLOR = (0, 0, 255)
OFFLINE_PROGRESS_COLOR = (255, 165, 0)

# --- Offline Processing ---
SHOW_OFFLINE_PROCESSING_PREVIEW = True
OFFLINE_PREVIEW_RESIZE_FACTOR = 1

# %% ===========================================================================
# VISUALIZATION HELPER CLASSES
# =============================================================================
class InfoPanelDrawer:
    """Handles drawing the consolidated information panel on the video frame."""
    def __init__(self, panel_width, font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=OVERLAY_FONT_SCALE, thickness=OVERLAY_THICKNESS):
        self.panel_width = panel_width
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.y_pos = 0
        self.x_margin = 20
        self.y_step = int(35 * self.font_scale)
        self.y_header_step = int(45 * self.font_scale)
        self.frame = None
        self.colors = {
            'white': (255, 255, 255), 'green': (100, 255, 100),
            'red': (100, 100, 255), 'yellow': (100, 230, 230),
            'cyan': (255, 255, 100), 'orange': (0, 165, 255)
        }
    def _reset_y(self):
        self.y_pos = self.y_header_step
    def _draw_text(self, text, color, is_header=False, indent=0):
        if is_header:
            self.y_pos += int(self.y_header_step * 0.6)
            cv2.putText(self.frame, text, (self.x_margin + indent, self.y_pos), self.font, self.font_scale * 1.1, self.colors['white'], self.thickness + 1)
            self.y_pos += int(self.y_step * 0.4)
            cv2.line(self.frame, (self.x_margin, self.y_pos), (self.panel_width - self.x_margin, self.y_pos), self.colors['white'], 1)
            self.y_pos += self.y_step
        else:
            cv2.putText(self.frame, text, (self.x_margin + indent, self.y_pos), self.font, self.font_scale, color, self.thickness)
            self.y_pos += self.y_step
    def draw(self, frame, proc_fps, cam_res, cam_fps, rec_status, ref_status, buoy_info, is_live=True):
        self.frame = frame
        panel = self.frame[:, :self.panel_width]
        cv2.addWeighted(panel, 0.5, np.zeros_like(panel), 0.5, 0, panel)
        self._reset_y()
        self._draw_text("PERFORMANCE", self.colors['white'], is_header=True)
        self._draw_text(f"Processing FPS: {proc_fps:.1f}", self.colors['cyan'])
        if is_live:
            self._draw_text(f"Camera: {cam_res[0]}x{cam_res[1]} @ {cam_fps:.1f} FPS", self.colors['cyan'])
        else:
            self._draw_text(f"Video Res: {cam_res[0]}x{cam_res[1]}", self.colors['cyan'])
        self._draw_text("SYSTEM STATUS", self.colors['white'], is_header=True)
        rec_on, rec_file = rec_status
        rec_file_basename = os.path.basename(rec_file) if rec_file else ""
        self._draw_text(f"Recording: {'ON' if rec_on else 'OFF'}", self.colors['green'] if rec_on else self.colors['red'])
        if rec_on and rec_file_basename:
            self._draw_text(rec_file_basename, self.colors['orange'], indent=20)
        ref_on, ref_found, ref_id = ref_status
        self._draw_text(f"Reference: {'ENABLED' if ref_on else 'DISABLED'}", self.colors['green'] if ref_on else self.colors['red'])
        if ref_on:
            self._draw_text(f"  > ID {ref_id}: {'Found' if ref_found else 'Lost'}", self.colors['green'] if ref_found else self.colors['red'], indent=20)
        self._draw_text("BUOY DETECTION", self.colors['white'], is_header=True)
        if not buoy_info:
            self._draw_text("None", self.colors['yellow'])
        else:
            sorted_buoys = sorted(buoy_info.items())
            for buoy_id, info in sorted_buoys:
                num_markers = info.get('num_markers', 0)
                quality_color = self.colors['green'] if num_markers >= 3 else (self.colors['yellow'] if num_markers == 2 else self.colors['red'])
                circle_center = (self.x_margin + 12, self.y_pos - 12)
                cv2.circle(self.frame, circle_center, 10, quality_color, -1)
                cv2.circle(self.frame, circle_center, 10, self.colors['white'], self.thickness)
                self._draw_text(f"Buoy {buoy_id}: {num_markers} marker(s)", self.colors['white'], indent=40)

class RealTimePlotter:
    """Handles creating and updating real-time Matplotlib plots for pose data."""
    def __init__(self, history_len, plot_width_px, video_height_px, buoy_ids_to_plot, is_relative):
        self.history_len = history_len
        self.plot_width_px = plot_width_px
        self.plot_height_px = video_height_px
        self.dpi = 100
        fig_width_in = self.plot_width_px / self.dpi
        fig_height_in = self.plot_height_px / self.dpi
        self.fig, self.axes = plt.subplots(2, 1, figsize=(fig_width_in, fig_height_in), dpi=self.dpi, facecolor='#DDDDDD')
        self.ax_pos, self.ax_rot = self.axes
        self.start_time = None
        self.buoy_data = {}
        self.buoy_lines = {}
        cmap = matplotlib.colormaps['tab10']
        colors = cmap(np.linspace(0, 1, NUM_BUOYS))
        pos_unit = "m (Rel)" if is_relative else "m (Abs)"
        self.ax_pos.set_title("Buoy CoG Position", fontsize=PLOT_TITLE_FONTSIZE)
        self.ax_pos.set_ylabel(f"Position [{pos_unit}]", fontsize=PLOT_LABEL_FONTSIZE)
        self.ax_pos.grid(True, linestyle='--', alpha=0.6)
        self.ax_pos.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
        self.ax_rot.set_title("Buoy CoG Orientation (Stable Angles)", fontsize=PLOT_TITLE_FONTSIZE)
        self.ax_rot.set_ylabel("Rotation [deg]", fontsize=PLOT_LABEL_FONTSIZE)
        self.ax_rot.set_xlabel("Time [s]", fontsize=PLOT_LABEL_FONTSIZE)
        self.ax_rot.grid(True, linestyle='--', alpha=0.6)
        self.ax_rot.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
        for i, buoy_id in enumerate(buoy_ids_to_plot):
            color = colors[i]
            label = f"B{buoy_id}"
            self.buoy_data[buoy_id] = {'time': deque(maxlen=history_len), 'pos_x': deque(maxlen=history_len), 'pos_y': deque(maxlen=history_len), 'pos_z': deque(maxlen=history_len), 'rot_x': deque(maxlen=history_len), 'rot_y': deque(maxlen=history_len), 'rot_z': deque(maxlen=history_len)}
            pos_lines = {'x': self.ax_pos.plot([], [], label=f'{label} X', color=color, linestyle='-', linewidth=PLOT_LINE_WIDTH)[0], 'y': self.ax_pos.plot([], [], label=f'{label} Y', color=color, linestyle='--', linewidth=PLOT_LINE_WIDTH)[0], 'z': self.ax_pos.plot([], [], label=f'{label} Z', color=color, linestyle=':', linewidth=PLOT_LINE_WIDTH)[0]}
            rot_lines = {'x': self.ax_rot.plot([], [], label=f'{label} Pitch', color=color, linestyle='-', linewidth=PLOT_LINE_WIDTH)[0], 'y': self.ax_rot.plot([], [], label=f'{label} Inclination', color=color, linestyle='--', linewidth=PLOT_LINE_WIDTH)[0], 'z': self.ax_rot.plot([], [], label=f'{label} Yaw', color=color, linestyle=':', linewidth=PLOT_LINE_WIDTH)[0]}
            self.buoy_lines[buoy_id] = {'pos': pos_lines, 'rot': rot_lines}
        self.ax_pos.legend(loc='upper left', fontsize=PLOT_LEGEND_FONTSIZE, ncol=max(1, len(buoy_ids_to_plot)))
        self.ax_rot.legend(loc='upper left', fontsize=PLOT_LEGEND_FONTSIZE, ncol=max(1, len(buoy_ids_to_plot)))
        self.fig.tight_layout(pad=3.5)
    def _canvas_to_numpy(self):
        self.fig.canvas.draw()
        buf = self.fig.canvas.tostring_rgb()
        ncols, nrows = self.fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    def update_and_get_plot_image(self, new_data_dict, timestamp):
        if self.start_time is None:
            self.start_time = timestamp
        current_rel_time = timestamp - self.start_time
        for buoy_id, data in new_data_dict.items():
            if buoy_id not in self.buoy_data: continue
            self.buoy_data[buoy_id]['time'].append(current_rel_time)
            self.buoy_data[buoy_id]['pos_x'].append(data['pos'][0])
            self.buoy_data[buoy_id]['pos_y'].append(data['pos'][1])
            self.buoy_data[buoy_id]['pos_z'].append(data['pos'][2])
            stable_rot = data['rot']
            self.buoy_data[buoy_id]['rot_x'].append(stable_rot[0])
            self.buoy_data[buoy_id]['rot_y'].append(stable_rot[1])
            self.buoy_data[buoy_id]['rot_z'].append(stable_rot[2])
        for buoy_id, lines_dict in self.buoy_lines.items():
            time_data = self.buoy_data[buoy_id]['time']
            if not time_data: continue
            lines_dict['pos']['x'].set_data(time_data, self.buoy_data[buoy_id]['pos_x'])
            lines_dict['pos']['y'].set_data(time_data, self.buoy_data[buoy_id]['pos_y'])
            lines_dict['pos']['z'].set_data(time_data, self.buoy_data[buoy_id]['pos_z'])
            lines_dict['rot']['x'].set_data(time_data, self.buoy_data[buoy_id]['rot_x'])
            lines_dict['rot']['y'].set_data(time_data, self.buoy_data[buoy_id]['rot_y'])
            lines_dict['rot']['z'].set_data(time_data, self.buoy_data[buoy_id]['rot_z'])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        return self._canvas_to_numpy()

# %% ===========================================================================
# UTILITY AND CORE MATHEMATICAL FUNCTIONS
# =============================================================================
def save_settings():
    """Saves the current application settings to a JSON file."""
    settings_to_save = {
        # Camera Setup
        "CAMERA_INDEX": CAMERA_INDEX,
        "CAMERA_RESOLUTION": CAMERA_RESOLUTION,
        "CAMERA_FPS": CAMERA_FPS,
        "REVERSE_CAMERA_DISPLAY": REVERSE_CAMERA_DISPLAY,
        # Visualization
        "SHOW_REALTIME_PLOTS": SHOW_REALTIME_PLOTS,
        # Video Recording
        "RECORD_VIDEO": RECORD_VIDEO,
        # ArUco Marker Configuration
        "MARKER_SIZE_METERS": MARKER_SIZE_METERS,
        # Reference Marker Configuration
        "REFERENCE_MARKER_ID": REFERENCE_MARKER_ID,
        "USE_REFERENCE_MARKER": USE_REFERENCE_MARKER,
        # LED Sync Configuration
        "ENABLE_LED_DETECTION": ENABLE_LED_DETECTION,
        "LED_DETECTION_ROI": LED_DETECTION_ROI,
        "LED_OFF_BASELINE_BRIGHTNESS": LED_OFF_BASELINE_BRIGHTNESS,
        "LED_BRIGHTNESS_THRESHOLD": LED_BRIGHTNESS_THRESHOLD,
        # Logging
        "LOG_ALL_MARKER_DATA": LOG_ALL_MARKER_DATA
    }
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, SETTINGS_FILE)
    try:
        with open(filepath, 'w') as f:
            json.dump(settings_to_save, f, indent=4)
        print(f"Settings successfully saved to {filepath}")
    except Exception as e:
        print(f"ERROR: Could not save settings to {filepath}. Reason: {e}")

def load_settings():
    """Loads application settings from a JSON file, overwriting defaults."""
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, SETTINGS_FILE)
    if not os.path.exists(filepath):
        print(f"INFO: Settings file not found at {filepath}. Using default values.")
        return
    try:
        with open(filepath, 'r') as f:
            settings = json.load(f)
        # Use global declarations to modify the module-level variables
        global CAMERA_INDEX, CAMERA_RESOLUTION, CAMERA_FPS, REVERSE_CAMERA_DISPLAY
        global SHOW_REALTIME_PLOTS, RECORD_VIDEO, MARKER_SIZE_METERS
        global REFERENCE_MARKER_ID, USE_REFERENCE_MARKER
        global ENABLE_LED_DETECTION, LED_DETECTION_ROI, LED_OFF_BASELINE_BRIGHTNESS
        global LED_BRIGHTNESS_THRESHOLD, LOG_ALL_MARKER_DATA

        # Update globals, using .get() to avoid errors if a key is missing
        CAMERA_INDEX = settings.get("CAMERA_INDEX", CAMERA_INDEX)
        CAMERA_RESOLUTION = tuple(settings.get("CAMERA_RESOLUTION", CAMERA_RESOLUTION))
        CAMERA_FPS = settings.get("CAMERA_FPS", CAMERA_FPS)
        REVERSE_CAMERA_DISPLAY = settings.get("REVERSE_CAMERA_DISPLAY", REVERSE_CAMERA_DISPLAY)
        SHOW_REALTIME_PLOTS = settings.get("SHOW_REALTIME_PLOTS", SHOW_REALTIME_PLOTS)
        RECORD_VIDEO = settings.get("RECORD_VIDEO", RECORD_VIDEO)
        MARKER_SIZE_METERS = settings.get("MARKER_SIZE_METERS", MARKER_SIZE_METERS)
        REFERENCE_MARKER_ID = settings.get("REFERENCE_MARKER_ID", REFERENCE_MARKER_ID)
        USE_REFERENCE_MARKER = settings.get("USE_REFERENCE_MARKER", USE_REFERENCE_MARKER)
        ENABLE_LED_DETECTION = settings.get("ENABLE_LED_DETECTION", ENABLE_LED_DETECTION)
        LED_DETECTION_ROI = settings.get("LED_DETECTION_ROI", LED_DETECTION_ROI)
        if LED_DETECTION_ROI: LED_DETECTION_ROI = tuple(LED_DETECTION_ROI)
        LED_OFF_BASELINE_BRIGHTNESS = settings.get("LED_OFF_BASELINE_BRIGHTNESS", LED_OFF_BASELINE_BRIGHTNESS)
        LED_BRIGHTNESS_THRESHOLD = settings.get("LED_BRIGHTNESS_THRESHOLD", LED_BRIGHTNESS_THRESHOLD)
        LOG_ALL_MARKER_DATA = settings.get("LOG_ALL_MARKER_DATA", LOG_ALL_MARKER_DATA)
        print(f"Settings loaded successfully from {filepath}")
    except (json.JSONDecodeError, TypeError) as e:
        print(f"ERROR: Could not load or parse settings file {filepath}. Using defaults. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading settings: {e}")

def load_camera_calibration():
    calibration_path = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, CALIBRATION_FILE)
    try:
        data = np.load(calibration_path)
        return data['camera_matrix'], data['dist_coeffs']
    except Exception:
        print(f"WARN: Calibration file not found or error loading. Using defaults: {calibration_path}")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

def get_video_writer(frame_size, full_video_path, fps=None):
    output_fps = fps if fps is not None and fps > 0 else VIDEO_FPS
    codec_to_try = VIDEO_CODEC
    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_to_try)
        writer = cv2.VideoWriter(full_video_path, fourcc, output_fps, frame_size)
        if writer.isOpened():
            print(f"Video writer created for: {full_video_path} with codec {codec_to_try}.")
            return writer
    except Exception as e:
        print(f"ERROR: Initializing writer with codec {codec_to_try}: {e}")
    print("Trying fallback codec 'XVID'.")
    try:
        fourcc_xvid = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(full_video_path, fourcc_xvid, output_fps, frame_size)
        if writer.isOpened():
            print(f"Video writer created with fallback codec 'XVID'.")
            return writer
    except Exception as e:
        print(f"ERROR: Initializing writer with fallback codec 'XVID': {e}")
    print(f"FATAL: Failed to create video writer for: {full_video_path}")
    return None

def load_buoy_geometry_calibrations():
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, BUOY_GEOMETRY_CALIBRATION_FILE)
    if not os.path.exists(filepath):
        print(f"INFO: Buoy geometry calibration file not found: {filepath}. Using defaults.")
        return {}
    try:
        with open(filepath, 'r') as f:
            calibrations = json.load(f)
        for buoy_id_str, markers_data in calibrations.items():
            for marker_id_str, pose_data in markers_data.items():
                if 't_rel_to_X0' in pose_data and pose_data['t_rel_to_X0'] is not None:
                    pose_data['t_rel_to_X0'] = np.array(pose_data['t_rel_to_X0'], dtype=np.float32)
                if 'rvec_rel_to_X0' in pose_data and pose_data['rvec_rel_to_X0'] is not None:
                    pose_data['rvec_rel_to_X0'] = np.array(pose_data['rvec_rel_to_X0'], dtype=np.float32)
        print(f"Loaded buoy geometry calibrations from: {filepath}")
        return calibrations
    except Exception as e:
        print(f"ERROR: Loading buoy geometry calibrations from {filepath}: {e}. Using defaults.")
        return {}

def save_buoy_geometry_calibrations(calibrations):
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, BUOY_GEOMETRY_CALIBRATION_FILE)
    serializable_calibrations = {}
    try:
        for buoy_id, markers_data in calibrations.items():
            serializable_calibrations[str(buoy_id)] = {}
            for marker_id, pose_data in markers_data.items():
                s_marker_data = {}
                if 't_rel_to_X0' in pose_data and isinstance(pose_data['t_rel_to_X0'], np.ndarray):
                    s_marker_data['t_rel_to_X0'] = pose_data['t_rel_to_X0'].tolist()
                else: s_marker_data['t_rel_to_X0'] = None
                if 'rvec_rel_to_X0' in pose_data and isinstance(pose_data['rvec_rel_to_X0'], np.ndarray):
                    s_marker_data['rvec_rel_to_X0'] = pose_data['rvec_rel_to_X0'].tolist()
                else: s_marker_data['rvec_rel_to_X0'] = None
                serializable_calibrations[str(buoy_id)][str(marker_id)] = s_marker_data
        with open(filepath, 'w') as f: json.dump(serializable_calibrations, f, indent=4)
        print(f"Saved buoy geometry calibrations to: {filepath}")
    except Exception as e:
        print(f"ERROR: Saving buoy geometry calibrations to {filepath}: {e}")

ALL_BUOY_GEOMETRY_CALIBRATIONS = {}
MARKER_BUOY_GEOMETRY_CONFIG = {}
BUOY_COG_OFFSETS_IN_BRF_CONFIG = {}
def initialize_buoy_configs():
    global MARKER_BUOY_GEOMETRY_CONFIG, BUOY_COG_OFFSETS_IN_BRF_CONFIG, ALL_BUOY_GEOMETRY_CALIBRATIONS
    ALL_BUOY_GEOMETRY_CALIBRATIONS = load_buoy_geometry_calibrations()
    temp_marker_buoy_geometry_config = {}
    for buoy_id, marker_ids in BUOY_ID_TO_MARKER_IDS_MAP.items():
        buoy_geom = {}
        buoy_id_str = str(buoy_id)
        id_X0 = marker_ids[0]
        buoy_geom[id_X0] = {'t_BRF_Xi': np.zeros((3, 1), dtype=np.float32), 'R_BRF_Xi': np.eye(3, dtype=np.float32)}
        for i_role, marker_id_actual in enumerate(marker_ids):
            if i_role == 0: continue
            marker_id_str = str(marker_id_actual)
            use_calibrated = False
            if buoy_id_str in ALL_BUOY_GEOMETRY_CALIBRATIONS and marker_id_str in ALL_BUOY_GEOMETRY_CALIBRATIONS[buoy_id_str]:
                calib_data = ALL_BUOY_GEOMETRY_CALIBRATIONS[buoy_id_str][marker_id_str]
                if calib_data and calib_data.get('t_rel_to_X0') is not None and calib_data.get('rvec_rel_to_X0') is not None:
                    use_calibrated = True
            if use_calibrated:
                calib_data = ALL_BUOY_GEOMETRY_CALIBRATIONS[buoy_id_str][marker_id_str]
                t_calib = np.array(calib_data['t_rel_to_X0'], dtype=np.float32).reshape(3, 1)
                R_calib, _ = cv2.Rodrigues(np.array(calib_data['rvec_rel_to_X0'], dtype=np.float32))
                buoy_geom[marker_id_actual] = {'t_BRF_Xi': t_calib, 'R_BRF_Xi': R_calib}
            elif i_role in MARKER_RELATIVE_GEOMETRY:
                ideal_geom = MARKER_RELATIVE_GEOMETRY[i_role]
                R_ideal, _ = cv2.Rodrigues(ideal_geom['rel_rvec_to_X0'])
                buoy_geom[marker_id_actual] = {'t_BRF_Xi': ideal_geom['rel_tvec_to_X0'].reshape(3, 1), 'R_BRF_Xi': R_ideal}
            else:
                buoy_geom[marker_id_actual] = {'t_BRF_Xi': np.full((3, 1), np.nan), 'R_BRF_Xi': np.full((3, 3), np.nan)}
        temp_marker_buoy_geometry_config[buoy_id] = buoy_geom
    MARKER_BUOY_GEOMETRY_CONFIG.clear()
    MARKER_BUOY_GEOMETRY_CONFIG.update(temp_marker_buoy_geometry_config)
    BUOY_COG_OFFSETS_IN_BRF_CONFIG.clear()
    BUOY_COG_OFFSETS_IN_BRF_CONFIG.update({buoy_id: BUOY_COG_OFFSET_VECTOR_ASSEMBLY_FRAME.reshape(3, 1) for buoy_id in BUOY_ID_TO_MARKER_IDS_MAP.keys()})
    print("Buoy configurations initialized/re-initialized.")

initialize_buoy_configs()

def rotation_vector_to_euler(rvec):
    if rvec is None: return np.zeros(3), R.identity().as_quat()
    rvec_np = np.asarray(rvec).reshape(3)
    if np.allclose(rvec_np, 0): return np.zeros(3), R.identity().as_quat()
    try:
        rotation = R.from_rotvec(rvec_np)
        euler_angles_deg = rotation.as_euler('xyz', degrees=True)
        quaternion = rotation.as_quat()
        return euler_angles_deg, quaternion
    except Exception: return np.zeros(3), R.identity().as_quat()

def get_stable_orientation_angles_from_matrix(rot_matrix):
    body_x_axis = np.array([1., 0., 0.])
    body_z_axis = np.array([0., 0., 1.])
    body_x_in_world = rot_matrix @ body_x_axis
    body_z_in_world = rot_matrix @ body_z_axis
    yaw = np.rad2deg(np.arctan2(body_x_in_world[1], body_x_in_world[0]))
    pitch = np.rad2deg(np.arcsin(-np.clip(body_x_in_world[2], -1.0, 1.0)))
    cos_inclination = np.clip(body_z_in_world[2], -1.0, 1.0)
    inclination = np.rad2deg(np.arccos(cos_inclination))
    return yaw, pitch, inclination

def transform_pose_to_reference(rvec_obj_cam, tvec_obj_cam, rvec_ref_cam, tvec_ref_cam):
    R_obj_cam, _ = cv2.Rodrigues(np.asarray(rvec_obj_cam))
    T_cam_obj = np.eye(4); T_cam_obj[:3, :3] = R_obj_cam; T_cam_obj[:3, 3] = np.asarray(tvec_obj_cam).flatten()
    R_ref_cam, _ = cv2.Rodrigues(np.asarray(rvec_ref_cam))
    T_cam_ref = np.eye(4); T_cam_ref[:3, :3] = R_ref_cam; T_cam_ref[:3, 3] = np.asarray(tvec_ref_cam).flatten()
    T_ref_cam = np.linalg.inv(T_cam_ref)
    T_ref_obj = T_ref_cam @ T_cam_obj
    R_ref_obj = T_ref_obj[:3, :3]
    t_ref_obj = T_ref_obj[:3, 3]
    rvec_ref_obj, _ = cv2.Rodrigues(R_ref_obj)
    return rvec_ref_obj.reshape(3, 1), t_ref_obj.reshape(3, 1)

def average_quaternions_weighted(quaternions, weights):
    if not quaternions: return R.identity().as_quat()
    if len(quaternions) == 1: return quaternions[0]
    q_ref = quaternions[0]
    for i in range(1, len(quaternions)):
        if np.dot(q_ref, quaternions[i]) < 0: quaternions[i] *= -1
    M = np.zeros((4, 4))
    for q_i, w_i in zip(quaternions, weights):
        q_i = q_i.reshape(4, 1)
        M += w_i * (q_i @ q_i.T)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
    return avg_quat / np.linalg.norm(avg_quat)

# %% ===========================================================================
# CORE FRAME PROCESSING LOGIC
# =============================================================================
def _process_frame_common(
    gray_frame, color_frame_to_annotate, camera_matrix, dist_coeffs, aruco_dict, aruco_parameters,
    marker_size_m, use_reference_marker_flag, reference_marker_id_val,
    positions_history,
    log_date_str, log_time_str, timestamp,
    buoy_id_to_marker_ids_map_param,
    marker_buoy_geometry_config_param,
    buoy_cog_offsets_in_brf_config_param,
    log_all_marker_data_flag,
    plotter=None,
    enable_led_detection_flag=False,
    led_detection_roi_tuple=None,
    led_off_baseline=0.0,
    led_brightness_threshold=30.0
):
    cog_log_entries = []
    individual_marker_log_entries = []
    plot_data_for_frame = {}
    detected_buoys_summary = {}
    annotated_frame = color_frame_to_annotate
    
    led_state_this_frame = False
    if enable_led_detection_flag and led_detection_roi_tuple is not None:
        try:
            x, y, w, h = [int(v) for v in led_detection_roi_tuple]
            if y + h <= color_frame_to_annotate.shape[0] and x + w <= color_frame_to_annotate.shape[1] and w > 0 and h > 0:
                led_roi = color_frame_to_annotate[y:y+h, x:x+w]
                current_red_mean = np.mean(led_roi[:, :, 2])
                if current_red_mean > (led_off_baseline + led_brightness_threshold):
                    led_state_this_frame = True
                roi_color = (0, 255, 0) if led_state_this_frame else (128, 128, 128)
                status_text = "LED: ON" if led_state_this_frame else "LED: OFF"
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), roi_color, 2)
                cv2.putText(annotated_frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.7, roi_color, OVERLAY_THICKNESS)
        except Exception as e:
            print(f"Warning: Error during LED detection: {e}")

    corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)
    ref_marker_detected_this_frame, ref_rvec, ref_tvec = False, None, None
    all_markers_data_this_frame = {}

    if ids is not None and len(ids) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners_list = [cv2.cornerSubPix(gray_frame, c.astype(np.float32), (5,5), (-1,-1), criteria) for c in corners]
        aruco.drawDetectedMarkers(annotated_frame, refined_corners_list, ids)
        rvecs_all, tvecs_all, _ = aruco.estimatePoseSingleMarkers(refined_corners_list, marker_size_m, camera_matrix, dist_coeffs)

        if use_reference_marker_flag:
            try:
                ref_idx = np.where(ids.flatten() == reference_marker_id_val)[0][0]
                ref_marker_detected_this_frame = True
                ref_rvec, ref_tvec = rvecs_all[ref_idx].reshape(3,1), tvecs_all[ref_idx].reshape(3,1)
                cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs, ref_rvec, ref_tvec, marker_size_m, AXES_THICKNESS)
            except IndexError: pass

        for i, marker_id in enumerate(ids.flatten()):
            r_cam_marker, t_cam_marker = rvecs_all[i].reshape(3,1), tvecs_all[i].reshape(3,1)
            _, quat_cam_marker = rotation_vector_to_euler(r_cam_marker)
            marker_data = {'cam_tvec': t_cam_marker, 'cam_rvec': r_cam_marker, 'cam_quat': quat_cam_marker, 'corners': refined_corners_list[i]}
            if use_reference_marker_flag and ref_marker_detected_this_frame:
                try:
                    r_rel, t_rel = transform_pose_to_reference(r_cam_marker, t_cam_marker, ref_rvec, ref_tvec)
                    _, quat_rel = rotation_vector_to_euler(r_rel)
                    marker_data['rel_tvec'] = t_rel; marker_data['rel_rvec'] = r_rel; marker_data['rel_quat'] = quat_rel
                except Exception: pass
            all_markers_data_this_frame[marker_id] = marker_data

        if log_all_marker_data_flag:
            for marker_id, data in all_markers_data_this_frame.items():
                log_row = [log_date_str, log_time_str, marker_id]
                cam_euler, _ = rotation_vector_to_euler(data['cam_rvec'])
                log_row.extend([f"{v:.6f}" for v in data['cam_tvec'].flatten()]); log_row.extend([f"{v:.3f}" for v in cam_euler.flatten()]); log_row.extend([f"{v:.6f}" for v in data['cam_quat']])
                if use_reference_marker_flag:
                    if 'rel_tvec' in data:
                        rel_euler, _ = rotation_vector_to_euler(data['rel_rvec'])
                        log_row.extend([f"{v:.6f}" for v in data['rel_tvec'].flatten()]); log_row.extend([f"{v:.3f}" for v in rel_euler.flatten()]); log_row.extend([f"{v:.6f}" for v in data['rel_quat']])
                    else: log_row.extend(["NaN"] * 10)
                individual_marker_log_entries.append(log_row)

        buoy_pose_estimates = {}
        marker_obj_pts = np.array([[-marker_size_m/2, marker_size_m/2, 0], [marker_size_m/2, marker_size_m/2, 0], [marker_size_m/2, -marker_size_m/2, 0], [-marker_size_m/2, -marker_size_m/2, 0]], dtype=np.float32)

        for marker_id, data in all_markers_data_this_frame.items():
            buoy_id = MARKER_ID_TO_BUOY_ID_MAP.get(marker_id)
            if buoy_id is None: continue
            buoy_geom_map = marker_buoy_geometry_config_param.get(buoy_id)
            if not buoy_geom_map or marker_id not in buoy_geom_map: continue
            geom_Xi_in_BRF = buoy_geom_map[marker_id]
            if np.isnan(geom_Xi_in_BRF['t_BRF_Xi']).any(): continue
            t_BRF_Xi, R_BRF_Xi = geom_Xi_in_BRF['t_BRF_Xi'], geom_Xi_in_BRF['R_BRF_Xi']
            R_cam_Xi, _ = cv2.Rodrigues(data['cam_rvec'])
            R_Xi_BRF = R_BRF_Xi.T; t_Xi_BRF = -R_Xi_BRF @ t_BRF_Xi
            R_cam_BRF_est = R_cam_Xi @ R_Xi_BRF
            t_cam_BRF_est = (R_cam_Xi @ t_Xi_BRF) + data['cam_tvec']
            projected_corners, _ = cv2.projectPoints(marker_obj_pts, data['cam_rvec'], data['cam_tvec'], camera_matrix, dist_coeffs)
            error = cv2.norm(data['corners'].reshape(-1,2), projected_corners.reshape(-1,2), cv2.NORM_L2)
            confidence = 1.0 / (1.0 + error**2 + 1e-9)
            buoy_pose_estimates.setdefault(buoy_id, []).append({'R_est': R_cam_BRF_est, 't_est': t_cam_BRF_est, 'confidence': confidence})
            
        for buoy_id, estimates in buoy_pose_estimates.items():
            if not estimates: continue
            detected_buoys_summary[buoy_id] = {'num_markers': len(estimates)}
            tvecs_to_avg = [est['t_est'] for est in estimates]
            quats_to_avg = [R.from_matrix(est['R_est']).as_quat() for est in estimates]
            weights = [est['confidence'] for est in estimates]
            tvecs_np = np.hstack(tvecs_to_avg)
            weights_np = np.array(weights).reshape(1, -1)
            fused_t_cam_BRF = (np.sum(tvecs_np * weights_np, axis=1) / np.sum(weights_np)).reshape(3,1)
            fused_quat_cam_BRF = average_quaternions_weighted(quats_to_avg, weights)
            fused_R_cam_BRF = R.from_quat(fused_quat_cam_BRF).as_matrix()
            cog_offset_in_brf = buoy_cog_offsets_in_brf_config_param.get(buoy_id, np.zeros((3,1)))
            t_cog_cam = fused_t_cam_BRF + (fused_R_cam_BRF @ cog_offset_in_brf)
            R_cog_cam = fused_R_cam_BRF; r_cog_cam, _ = cv2.Rodrigues(R_cog_cam)
            final_R_for_log = R_cog_cam; final_t_for_log = t_cog_cam; is_relative_log = False
            
            if use_reference_marker_flag and ref_marker_detected_this_frame:
                try:
                    R_ref_cam, _ = cv2.Rodrigues(ref_rvec)
                    final_R_for_log = R_ref_cam.T @ R_cog_cam
                    final_t_for_log = R_ref_cam.T @ (t_cog_cam - ref_tvec)
                    is_relative_log = True
                except Exception as e: print(f"Error transforming CoG to relative frame for buoy {buoy_id}: {e}")

            r_for_log, _ = cv2.Rodrigues(final_R_for_log)
            euler_for_log, quat_for_log = rotation_vector_to_euler(r_for_log)
            stable_angles = get_stable_orientation_angles_from_matrix(final_R_for_log)
            plot_data_for_frame[buoy_id] = {'pos': final_t_for_log.flatten(), 'rot': np.array([stable_angles[1], stable_angles[2], stable_angles[0]])}
            log_entry = [log_date_str, log_time_str, buoy_id, 1 if led_state_this_frame else 0]
            if use_reference_marker_flag:
                log_entry.append("1" if is_relative_log else "0")
                if is_relative_log:
                    log_entry.extend([f"{v:.6f}" for v in final_t_for_log.flatten()]); log_entry.extend([f"{v:.3f}" for v in euler_for_log.flatten()]); log_entry.extend([f"{v:.6f}" for v in quat_for_log])
                else: log_entry.extend(["NaN"] * 10)
            abs_euler, abs_quat = rotation_vector_to_euler(r_cog_cam)
            log_entry.extend([f"{v:.6f}" for v in t_cog_cam.flatten()]); log_entry.extend([f"{v:.3f}" for v in abs_euler.flatten()]); log_entry.extend([f"{v:.6f}" for v in abs_quat])
            cog_log_entries.append(log_entry)

            try:
                imgpts_cog, _ = cv2.projectPoints(np.zeros((1,3)), r_cog_cam, t_cog_cam, camera_matrix, dist_coeffs)
                center_px = tuple(imgpts_cog[0,0].astype(int))
                positions_history.setdefault(buoy_id, deque(maxlen=TRAJECTORY_LENGTH)).append(center_px)
                cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs, r_cog_cam, t_cog_cam, marker_size_m * 0.8, AXES_THICKNESS)
                prefix = "Rel" if is_relative_log else "Abs"; y_off, y_step = -80, int(35 * OVERLAY_FONT_SCALE)
                cv2.putText(annotated_frame, f"B{buoy_id} CoG", (center_px[0] + 20, center_px[1] + y_off), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, BUOY_COG_INFO_COLOR, OVERLAY_THICKNESS); y_off += y_step
                cv2.putText(annotated_frame, f"{prefix} P:({final_t_for_log[0,0]:.2f},{final_t_for_log[1,0]:.2f},{final_t_for_log[2,0]:.2f})", (center_px[0] + 20, center_px[1] + y_off), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.8, BUOY_COG_INFO_COLOR, OVERLAY_THICKNESS); y_off += y_step
                cv2.putText(annotated_frame, f"{prefix} R:({euler_for_log[0]:.1f},{euler_for_log[1]:.1f},{euler_for_log[2]:.1f})", (center_px[0] + 20, center_px[1] + y_off), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.8, BUOY_COG_INFO_COLOR, OVERLAY_THICKNESS)
            except Exception as e: print(f"Error during CoG visualization for buoy {buoy_id}: {e}")

    plot_image = plotter.update_and_get_plot_image(plot_data_for_frame, timestamp) if plotter else None
    for buoy_id_hist, pos_list_hist in positions_history.items():
        if len(pos_list_hist) > 1:
            pts = np.array(pos_list_hist, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR, thickness=TRAJECTORY_THICKNESS)

    return (annotated_frame, plot_image, cog_log_entries, individual_marker_log_entries, positions_history, detected_buoys_summary, ref_marker_detected_this_frame)

# %% ===========================================================================
# TOP-LEVEL APPLICATION MODES
# =============================================================================
def track_buoy_with_aruco_6dof(parent_tk_window=None):
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_live_{experiment_timestamp}"
    experiment_dir = os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created live experiment directory: {experiment_dir}")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
    if not cap.isOpened(): messagebox.showerror("Cam Err", f"No cam {CAMERA_INDEX}", parent=parent_tk_window); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1]); cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    w, h, cam_fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cam_mat, dist_c = load_camera_calibration()
    aruco_dict, params = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE), aruco.DetectorParameters()
    positions_history = {}
    info_panel_drawer = InfoPanelDrawer(panel_width=INFO_PANEL_WIDTH_PX)
    plotter = None; output_frame_size = (w, h)
    if SHOW_REALTIME_PLOTS:
        plotter = RealTimePlotter(PLOT_HISTORY_LENGTH, PLOT_WIDTH_PX, h, list(BUOY_ID_TO_MARKER_IDS_MAP.keys()), USE_REFERENCE_MARKER)
        output_frame_size = (w + PLOT_WIDTH_PX, h)
    cog_csv_fn = os.path.join(experiment_dir, "tracking_data_cog.csv")
    header_cog = ['Date', 'Time', 'Buoy_ID', 'LED_On']
    if USE_REFERENCE_MARKER: header_cog.extend(['Ref_Marker_Frame_Detected', 'CoG_Pos_X_Rel', 'CoG_Pos_Y_Rel', 'CoG_Pos_Z_Rel', 'CoG_Rot_X_Rel_deg', 'CoG_Rot_Y_Rel_deg', 'CoG_Rot_Z_Rel_deg', 'CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel'])
    header_cog.extend(['CoG_Pos_X_Cam', 'CoG_Pos_Y_Cam', 'CoG_Pos_Z_Cam', 'CoG_Rot_X_Cam_deg', 'CoG_Rot_Y_Cam_deg', 'CoG_Rot_Z_Cam_deg', 'CoG_Quat_X_Cam', 'CoG_Quat_Y_Cam', 'CoG_Quat_Z_Cam', 'CoG_Quat_W_Cam'])
    cog_csv_file = open(cog_csv_fn, 'w', newline=''); csv.writer(cog_csv_file).writerow(header_cog)
    markers_csv_file = None
    if LOG_ALL_MARKER_DATA:
        markers_csv_fn = os.path.join(experiment_dir, "tracking_data_markers.csv")
        header_markers = ['Log_Date', 'Log_Time', 'Marker_ID', 'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z', 'Cam_Rot_X_deg', 'Cam_Rot_Y_deg', 'Cam_Rot_Z_deg', 'Cam_Quat_X', 'Cam_Quat_Y', 'Cam_Quat_Z', 'Cam_Quat_W']
        if USE_REFERENCE_MARKER: header_markers.extend(['Rel_Pos_X', 'Rel_Pos_Y', 'Rel_Pos_Z', 'Rel_Rot_X_deg', 'Rel_Rot_Y_deg', 'Rel_Rot_Z_deg', 'Rel_Quat_X', 'Rel_Quat_Y', 'Rel_Quat_Z', 'Rel_Quat_W'])
        markers_csv_file = open(markers_csv_fn, 'w', newline=''); csv.writer(markers_csv_file).writerow(header_markers)
    is_recording, rev_disp = RECORD_VIDEO, REVERSE_CAMERA_DISPLAY
    vid_writer, recording_path = None, None
    frame_times = deque(maxlen=20)
    try:
        while True:
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now_dt = datetime.now()
            date_s, time_s = now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S.%f")[:-3]
            annotated_frame, plot_img, cog_logs, marker_logs, positions_history, buoy_summary, ref_detected = _process_frame_common(
                gray, frame.copy(), cam_mat, dist_c, aruco_dict, params, MARKER_SIZE_METERS,
                USE_REFERENCE_MARKER, REFERENCE_MARKER_ID, positions_history,
                date_s, time_s, time.time(), BUOY_ID_TO_MARKER_IDS_MAP,
                MARKER_BUOY_GEOMETRY_CONFIG, BUOY_COG_OFFSETS_IN_BRF_CONFIG,
                LOG_ALL_MARKER_DATA, plotter,
                enable_led_detection_flag=ENABLE_LED_DETECTION,
                led_detection_roi_tuple=LED_DETECTION_ROI,
                led_off_baseline=LED_OFF_BASELINE_BRIGHTNESS,
                led_brightness_threshold=LED_BRIGHTNESS_THRESHOLD
            )
            if cog_logs: csv.writer(cog_csv_file).writerows(cog_logs)
            if marker_logs and markers_csv_file: csv.writer(markers_csv_file).writerows(marker_logs)
            proc_fps = 1.0 / np.mean(frame_times) if len(frame_times) > 1 else 0.0
            info_panel_drawer.draw(annotated_frame, proc_fps, (w, h), cam_fps, (is_recording, recording_path), (USE_REFERENCE_MARKER, ref_detected, REFERENCE_MARKER_ID), buoy_summary, is_live=True)
            final_frame = annotated_frame
            if plotter and plot_img is not None:
                final_frame = np.full((h, w + PLOT_WIDTH_PX, 3), 40, np.uint8)
                final_frame[:, :w] = annotated_frame
                final_frame[:, w:] = cv2.resize(plot_img, (PLOT_WIDTH_PX, h))
            cv2.putText(final_frame, "Q: Quit | R: Reverse Disp | V: Record", (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0,165,255), OVERLAY_THICKNESS)
            display_frame = cv2.flip(final_frame, 1) if rev_disp else final_frame
            cv2.imshow('Buoy 6DOF Tracking (Live)', display_frame)
            if is_recording and vid_writer: vid_writer.write(final_frame)
            frame_times.append(time.perf_counter() - t_start)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): rev_disp = not rev_disp
            elif key == ord('v'):
                is_recording = not is_recording
                if is_recording and not vid_writer:
                    recording_path = os.path.join(experiment_dir, f"live_annotated_{experiment_timestamp}{VIDEO_EXTENSION}")
                    vid_writer = get_video_writer(output_frame_size, recording_path, fps=cam_fps)
                    if not vid_writer: is_recording = False; print("ERROR: Could not start recording.")
                    else: print(f"Recording started to {recording_path}")
                elif not is_recording and vid_writer:
                    vid_writer.release(); vid_writer = None
                    print(f"Recording stopped. Video saved to {recording_path}"); recording_path = None
    finally:
        if vid_writer: vid_writer.release()
        if cog_csv_file: cog_csv_file.close()
        if markers_csv_file: markers_csv_file.close()
        cap.release(); cv2.destroyAllWindows()
        messagebox.showinfo("Live Tracking Ended", f"Experiment data saved in:\n{experiment_dir}", parent=parent_tk_window)

def process_video_offline(input_video_path, use_ref_marker_override,
                          gui_status_label=None, gui_progress_bar=None,
                          video_start_date_str=None, video_start_time_str=None,
                          led_roi_override=None, led_baseline_override=None):
    parent_tk = gui_status_label.master if gui_status_label else None
    if not os.path.exists(input_video_path): messagebox.showerror("File Err", f"No vid: {input_video_path}", parent=parent_tk); return
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_offline_{os.path.splitext(os.path.basename(input_video_path))[0]}_{experiment_timestamp}"
    experiment_dir = os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    try: shutil.copy2(input_video_path, os.path.join(experiment_dir, f"original_{os.path.basename(input_video_path)}"))
    except Exception as e: print(f"Warning: Could not copy input video to experiment folder: {e}")
    cap = cv2.VideoCapture(input_video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if gui_progress_bar: gui_progress_bar["maximum"]= total_frames if total_frames > 0 else 100
    cam_mat, dist_c = load_camera_calibration()
    aruco_dict, params = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE), aruco.DetectorParameters()
    positions_history = {}
    info_panel_drawer = InfoPanelDrawer(panel_width=INFO_PANEL_WIDTH_PX)
    plotter = None; output_frame_size = (w, h)
    if SHOW_REALTIME_PLOTS:
        plotter = RealTimePlotter(PLOT_HISTORY_LENGTH, PLOT_WIDTH_PX, h, list(BUOY_ID_TO_MARKER_IDS_MAP.keys()), use_ref_marker_override)
        output_frame_size = (w + PLOT_WIDTH_PX, h)
    annotated_vid_path = os.path.join(experiment_dir, f"annotated_{experiment_timestamp}{VIDEO_EXTENSION}")
    vid_writer = get_video_writer(output_frame_size, annotated_vid_path, fps=fps)
    cog_csv_fn = os.path.join(experiment_dir, "tracking_data_cog.csv")
    header_cog = ['Date', 'Time', 'Buoy_ID', 'LED_On']
    if use_ref_marker_override: header_cog.extend(['Ref_Marker_Frame_Detected', 'CoG_Pos_X_Rel', 'CoG_Pos_Y_Rel', 'CoG_Pos_Z_Rel', 'CoG_Rot_X_Rel_deg', 'CoG_Rot_Y_Rel_deg', 'CoG_Rot_Z_Rel_deg', 'CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel'])
    header_cog.extend(['CoG_Pos_X_Cam', 'CoG_Pos_Y_Cam', 'CoG_Pos_Z_Cam', 'CoG_Rot_X_Cam_deg', 'CoG_Rot_Y_Cam_deg', 'CoG_Rot_Z_Cam_deg', 'CoG_Quat_X_Cam', 'CoG_Quat_Y_Cam', 'CoG_Quat_Z_Cam', 'CoG_Quat_W_Cam'])
    cog_csv_file = open(cog_csv_fn, 'w', newline=''); csv.writer(cog_csv_file).writerow(header_cog)
    markers_csv_file = None
    if LOG_ALL_MARKER_DATA:
        markers_csv_fn = os.path.join(experiment_dir, "tracking_data_markers.csv")
        header_markers = ['Log_Date', 'Log_Time', 'Marker_ID', 'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z', 'Cam_Rot_X_deg', 'Cam_Rot_Y_deg', 'Cam_Rot_Z_deg', 'Cam_Quat_X', 'Cam_Quat_Y', 'Cam_Quat_Z', 'Cam_Quat_W']
        if use_ref_marker_override: header_markers.extend(['Rel_Pos_X', 'Rel_Pos_Y', 'Rel_Pos_Z', 'Rel_Rot_X_deg', 'Rel_Rot_Y_deg', 'Rel_Rot_Z_deg', 'Rel_Quat_X', 'Rel_Quat_Y', 'Rel_Quat_Z', 'Rel_Quat_W'])
        markers_csv_file = open(markers_csv_fn, 'w', newline=''); csv.writer(markers_csv_file).writerow(header_markers)
    base_timestamp, timestamp_note = None, "Using relative video time."
    if video_start_date_str and video_start_time_str:
        try: base_timestamp = datetime.strptime(f"{video_start_date_str.strip()} {video_start_time_str.strip()}", "%Y-%m-%d %H:%M:%S"); timestamp_note = "Using user-provided start timestamp."
        except ValueError: messagebox.showwarning("Date/Time Parse Error", "Could not parse date/time. Using relative time.", parent=parent_tk); timestamp_note = "Fallback to relative video time due to parse error."
    frame_count, start_proc_time = 0, time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_sec = frame_msec / 1000.0
            if base_timestamp:
                current_dt = base_timestamp + timedelta(milliseconds=frame_msec)
                date_s, time_s = current_dt.strftime("%Y-%m-%d"), current_dt.strftime("%H:%M:%S.%f")[:-3]
            else:
                date_s = datetime.now().strftime("%Y-%m-%d")
                time_s = (datetime.min + timedelta(milliseconds=frame_msec)).strftime("%H:%M:%S.%f")[:-3]
            annotated_frame, plot_img, cog_logs, marker_logs, positions_history, buoy_summary, ref_detected = _process_frame_common(
                gray, frame.copy(), cam_mat, dist_c, aruco_dict, params, MARKER_SIZE_METERS,
                use_ref_marker_override, REFERENCE_MARKER_ID, positions_history,
                date_s, time_s, frame_sec, BUOY_ID_TO_MARKER_IDS_MAP,
                MARKER_BUOY_GEOMETRY_CONFIG, BUOY_COG_OFFSETS_IN_BRF_CONFIG,
                LOG_ALL_MARKER_DATA, plotter,
                enable_led_detection_flag=ENABLE_LED_DETECTION,
                led_detection_roi_tuple=led_roi_override,
                led_off_baseline=led_baseline_override if led_baseline_override is not None else 0.0,
                led_brightness_threshold=LED_BRIGHTNESS_THRESHOLD
            )
            if cog_logs: csv.writer(cog_csv_file).writerows(cog_logs)
            if marker_logs and markers_csv_file: csv.writer(markers_csv_file).writerows(marker_logs)
            proc_fps = frame_count / (time.time() - start_proc_time) if time.time() > start_proc_time else 0.0
            info_panel_drawer.draw(annotated_frame, proc_fps, (w, h), fps, (vid_writer is not None, annotated_vid_path), (use_ref_marker_override, ref_detected, REFERENCE_MARKER_ID), buoy_summary, is_live=False)
            final_frame = annotated_frame
            if plotter and plot_img is not None:
                final_frame = np.full((h, w + PLOT_WIDTH_PX, 3), 40, np.uint8)
                final_frame[:, :w] = annotated_frame
                final_frame[:, w:] = cv2.resize(plot_img, (PLOT_WIDTH_PX, h))
            prog_txt = f"Frame: {frame_count}/{total_frames}"
            (tw, th), _ = cv2.getTextSize(prog_txt, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OVERLAY_THICKNESS)
            cv2.putText(final_frame, prog_txt, (w - tw - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OFFLINE_PROGRESS_COLOR, OVERLAY_THICKNESS)
            if vid_writer: vid_writer.write(final_frame)
            if frame_count % 25 == 0 or frame_count == total_frames:
                if total_frames > 0:
                    progress_pct = (frame_count / total_frames) * 100
                    if gui_status_label: gui_status_label.config(text=f"Processing... {progress_pct:.1f}%")
                    if gui_progress_bar: gui_progress_bar["value"] = frame_count
                    if parent_tk: parent_tk.update_idletasks()
            if SHOW_OFFLINE_PROCESSING_PREVIEW:
                preview_size = (int(output_frame_size[0]*OFFLINE_PREVIEW_RESIZE_FACTOR), int(output_frame_size[1]*OFFLINE_PREVIEW_RESIZE_FACTOR))
                cv2.imshow('Offline Preview', cv2.resize(final_frame, preview_size))
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
    finally:
        if vid_writer: vid_writer.release()
        if cog_csv_file: cog_csv_file.close()
        if markers_csv_file: markers_csv_file.close()
        cap.release()
        if SHOW_OFFLINE_PROCESSING_PREVIEW: cv2.destroyAllWindows()
        saved_files_msg = f"CoG data in: {cog_csv_fn}"
        if LOG_ALL_MARKER_DATA: saved_files_msg += f"\nMarkers data in: {markers_csv_fn}"
        final_msg = (f"Offline processing complete. Data saved in experiment folder:\n{experiment_dir}\n\n{saved_files_msg}\n\nTimestamping note: {timestamp_note}")
        if gui_status_label: gui_status_label.config(text=f"Done. Data in: {experiment_name}")
        messagebox.showinfo("Processing Complete", final_msg, parent=parent_tk)

def calibrate_camera(parent_tk_window=None):
    print("Starting Camera Calibration process...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
    if not cap.isOpened(): messagebox.showerror("Camera Error", f"Cannot open camera {CAMERA_INDEX} for calibration.", parent=parent_tk_window); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    w_cal, h_cal = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rev_disp_calib = REVERSE_CAMERA_DISPLAY; chessboard_shape = CALIBRATION_CHESSBOARD_SHAPE; square_size_meters = CHESSBOARD_SQUARE_SIZE_MM / 1000.0
    objp = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32); objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2) * square_size_meters
    obj_points_list, img_points_list = [], []; gray_frame_shape = None
    cv2.namedWindow('Camera Calibration - Q to Finish & Calibrate', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret: print("Error: Cannot grab frame for calibration."); break
        gray_frame_calib = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray_frame_shape is None: gray_frame_shape = gray_frame_calib.shape[::-1]
        display_frame_calib = frame.copy()
        ret_corners, corners_found = cv2.findChessboardCorners(gray_frame_calib, chessboard_shape, None)
        text_color, status_text = ((0,0,255), "Aim at chessboard...")
        if ret_corners:
            text_color, status_text = ((0,255,0), "Chessboard detected! Press 'c' to capture.")
            cv2.drawChessboardCorners(display_frame_calib, chessboard_shape, corners_found, ret_corners)
        cv2.putText(display_frame_calib, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, text_color, OVERLAY_THICKNESS)
        cv2.putText(display_frame_calib, f"Captured: {len(obj_points_list)}", (20, h_cal - 40), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 0, 0), OVERLAY_THICKNESS)
        cv2.putText(display_frame_calib, f"RevDisp: {'ON' if rev_disp_calib else 'OFF'}(r)", (20, h_cal - 100), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 165, 255), OVERLAY_THICKNESS)
        if rev_disp_calib: display_frame_calib = cv2.flip(display_frame_calib, 1)
        cv2.imshow('Camera Calibration - Q to Finish & Calibrate', display_frame_calib)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'): rev_disp_calib = not rev_disp_calib
        elif key == ord('c') and ret_corners:
            criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray_frame_calib, corners_found, (11,11), (-1,-1), criteria_subpix)
            obj_points_list.append(objp); img_points_list.append(corners_refined)
            print(f"Calibration image captured ({len(obj_points_list)} total).")
            cv2.drawChessboardCorners(frame, chessboard_shape, corners_refined, True)
            temp_disp_capture_confirm = frame.copy()
            if rev_disp_calib: temp_disp_capture_confirm = cv2.flip(temp_disp_capture_confirm, 1)
            cv2.imshow('Camera Calibration - Q to Finish & Calibrate', temp_disp_capture_confirm); cv2.waitKey(500)
    cap.release(); cv2.destroyAllWindows()
    if len(obj_points_list) > 3 and gray_frame_shape is not None:
        print(f"Calculating camera calibration with {len(obj_points_list)} images...")
        ret_calib, camera_matrix_calib, dist_coeffs_calib, rvecs_calib, tvecs_calib = cv2.calibrateCamera(obj_points_list, img_points_list, gray_frame_shape, None, None)
        if ret_calib:
            mean_error = sum(cv2.norm(img_points_list[i], cv2.projectPoints(obj_points_list[i], rvecs_calib[i], tvecs_calib[i], camera_matrix_calib, dist_coeffs_calib)[0], cv2.NORM_L2)/len(imgpoints2) for i in range(len(obj_points_list))) / len(obj_points_list) if len(obj_points_list) > 0 else 0.0
            reproj_error_message = f"Mean Reprojection Error: {mean_error:.4f} pixels"
            print(f"Cam Matrix:\n{camera_matrix_calib}\nDist Coeffs:\n{dist_coeffs_calib.ravel()}\n{reproj_error_message}")
            calibration_filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, CALIBRATION_FILE)
            try:
                np.savez(calibration_filepath, camera_matrix=camera_matrix_calib, dist_coeffs=dist_coeffs_calib, reprojection_error=mean_error, num_images=len(obj_points_list))
                messagebox.showinfo("Calibration OK", f"Saved: '{calibration_filepath}'.\n{reproj_error_message}", parent=parent_tk_window)
            except Exception as e: messagebox.showerror("Save Error", f"Failed to save calib: {e}", parent=parent_tk_window)
        else: messagebox.showwarning("Calib Fail", "cv2.calibrateCamera false.", parent=parent_tk_window)
    else: messagebox.showwarning("Calib Fail", f"Need >3 images (got {len(obj_points_list)}).", parent=parent_tk_window)

def calibrate_buoy_marker_geometry(buoy_id_to_calibrate, parent_tk_window=None, num_frames_to_capture=30):
    global ALL_BUOY_GEOMETRY_CALIBRATIONS
    if buoy_id_to_calibrate not in BUOY_ID_TO_MARKER_IDS_MAP: messagebox.showerror("Error", f"Buoy ID {buoy_id_to_calibrate} not defined.", parent=parent_tk_window); return
    target_marker_ids = BUOY_ID_TO_MARKER_IDS_MAP[buoy_id_to_calibrate]
    if len(target_marker_ids) < 2: messagebox.showinfo("Info", f"Buoy ID {buoy_id_to_calibrate} has < 2 markers. No relative calibration needed.", parent=parent_tk_window); return
    id_X0 = target_marker_ids[0]; ids_to_calibrate_relative_to_X0 = target_marker_ids[1:]
    msg = (f"Starting calibration for Buoy ID: {buoy_id_to_calibrate}.\n" f"Reference (X0): ID {id_X0}\n" f"Markers to calibrate: {ids_to_calibrate_relative_to_X0}\n\n" f"Ensure X0 and at least one other target marker are visible together.\n" f"Press 'c' to capture (up to {num_frames_to_capture} total). Press 'q' to finish.")
    messagebox.showinfo("Buoy Calibration", msg, parent=parent_tk_window)
    cam_matrix, dist_coeffs = load_camera_calibration()
    aruco_dict_obj, parameters_obj = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE), aruco.DetectorParameters()
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
    if not cap.isOpened(): messagebox.showerror("Cam Error", f"No cam {CAMERA_INDEX}.", parent=parent_tk_window); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    collected_relative_poses = {marker_id: {'tvecs': [], 'rvecs': []} for marker_id in ids_to_calibrate_relative_to_X0}
    frames_captured_count = 0
    window_name = f"Buoy {buoy_id_to_calibrate} Calibration - 'c' capture, 'q' finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while frames_captured_count < num_frames_to_capture:
        ret, frame = cap.read()
        if not ret: messagebox.showerror("Cam Error", "Failed to grab frame.", parent=parent_tk_window); break
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict_obj, parameters=parameters_obj)
        poses_cam_this_frame = {}
        if ids is not None:
            aruco.drawDetectedMarkers(display_frame, corners, ids)
            rvecs_est, tvecs_est, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_METERS, cam_matrix, dist_coeffs)
            for i, marker_id_arr in enumerate(ids):
                marker_id = marker_id_arr[0]
                if marker_id in target_marker_ids:
                    poses_cam_this_frame[marker_id] = (rvecs_est[i].reshape(3,1), tvecs_est[i].reshape(3,1))
                    cv2.drawFrameAxes(display_frame, cam_matrix, dist_coeffs, rvecs_est[i], tvecs_est[i], MARKER_SIZE_METERS, AXES_THICKNESS//2)
        x0_is_visible = id_X0 in poses_cam_this_frame
        any_xi_visible_with_x0 = any(xi_id in poses_cam_this_frame for xi_id in ids_to_calibrate_relative_to_X0)
        status_text_calib, status_color_calib = "Aim: Show X0 + another buoy marker", (0, 165, 255)
        if x0_is_visible and any_xi_visible_with_x0: status_text_calib, status_color_calib = "OK: Press 'c'.", (0, 255, 0)
        elif x0_is_visible: status_text_calib, status_color_calib = f"Show another marker with X0.", (0, 255, 255)
        else: status_text_calib, status_color_calib = f"X0 (ID {id_X0}) NOT visible.", (0, 0, 255)
        cv2.putText(display_frame, status_text_calib, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, status_color_calib, OVERLAY_THICKNESS)
        cv2.putText(display_frame, f"Buoy ID: {buoy_id_to_calibrate} (X0: {id_X0})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 255, 0), OVERLAY_THICKNESS)
        cv2.putText(display_frame, f"Good 'c' Presses: {frames_captured_count}/{num_frames_to_capture}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 255, 255), OVERLAY_THICKNESS)
        y_offset_counts = 200
        for xi_id_disp in ids_to_calibrate_relative_to_X0:
            count_xi = len(collected_relative_poses[xi_id_disp]['tvecs'])
            cv2.putText(display_frame, f"ID {xi_id_disp} samples: {count_xi}", (20, y_offset_counts), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE*0.8, (200, 200, 50), OVERLAY_THICKNESS)
            y_offset_counts += 40
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            if not x0_is_visible: messagebox.showwarning("Capture Skipped", f"X0 not visible.", parent=parent_tk_window); continue
            rvec_X0_cam, tvec_X0_cam = poses_cam_this_frame[id_X0]
            captured_pair_this_press = False
            for marker_id_Xi in ids_to_calibrate_relative_to_X0:
                if marker_id_Xi in poses_cam_this_frame:
                    try:
                        rvec_Xi_cam, tvec_Xi_cam = poses_cam_this_frame[marker_id_Xi]
                        rvec_X0_Xi, tvec_X0_Xi = transform_pose_to_reference(rvec_Xi_cam, tvec_Xi_cam, rvec_X0_cam, tvec_X0_cam)
                        collected_relative_poses[marker_id_Xi]['tvecs'].append(tvec_X0_Xi.flatten())
                        collected_relative_poses[marker_id_Xi]['rvecs'].append(rvec_X0_Xi.flatten())
                        print(f"  Captured relative pose for X0 <-- Xi(ID {marker_id_Xi})")
                        captured_pair_this_press = True
                    except Exception as e: print(f"Error calculating relative pose: {e}")
            if captured_pair_this_press: frames_captured_count += 1
            else: messagebox.showinfo("Capture Info", "X0 was visible, but no other target markers were.", parent=parent_tk_window)
    cap.release(); cv2.destroyAllWindows()
    if not any(len(data['tvecs']) > 0 for data in collected_relative_poses.values()): messagebox.showwarning("Incomplete", "No valid data pairs (X0, Xi) captured.", parent=parent_tk_window); return
    low_sample_markers = [mid for mid, data in collected_relative_poses.items() if 0 < len(data['tvecs']) < BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR]
    if low_sample_markers and not messagebox.askyesno("Warning", f"Markers with <{BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR} samples: {low_sample_markers}. Continue?", icon='warning', parent=parent_tk_window): return
    print(f"\nCalculating avg poses for Buoy ID {buoy_id_to_calibrate}...")
    buoy_calib_results = {}
    for marker_id_Xi, data in collected_relative_poses.items():
        if not data['tvecs']: buoy_calib_results[str(marker_id_Xi)] = {'t_rel_to_X0': None, 'rvec_rel_to_X0': None}; continue
        avg_tvec = np.mean(np.array(data['tvecs']), axis=0)
        quats = [R.from_rotvec(rvec).as_quat() for rvec in data['rvecs']]
        avg_quat = average_quaternions_weighted(quats, [1.0] * len(quats))
        avg_rvec = R.from_quat(avg_quat).as_rotvec()
        print(f"  Marker {marker_id_Xi} -> tvec={avg_tvec}, rvec={avg_rvec}")
        buoy_calib_results[str(marker_id_Xi)] = {'t_rel_to_X0': avg_tvec.astype(np.float32), 'rvec_rel_to_X0': avg_rvec.astype(np.float32)}
    ALL_BUOY_GEOMETRY_CALIBRATIONS[str(buoy_id_to_calibrate)] = buoy_calib_results
    save_buoy_geometry_calibrations(ALL_BUOY_GEOMETRY_CALIBRATIONS)
    initialize_buoy_configs()
    messagebox.showinfo("Complete", f"Calibration for Buoy ID {buoy_id_to_calibrate} finished and saved.", parent=parent_tk_window)

def generate_aruco_marker(marker_id, size_px=600, border_px=50, save=True, parent_tk_window=None):
    aruco_dict_gen = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    try: img_gray = aruco.generateImageMarker(aruco_dict_gen, marker_id, size_px)
    except cv2.error as e: messagebox.showerror("Marker Gen Err", f"ID {marker_id}: {e}", parent=parent_tk_window); return
    img_bgr_bord = cv2.cvtColor(np.pad(img_gray, border_px, 'constant', constant_values=255), cv2.COLOR_GRAY2BGR)
    font, txt_col, tot_sz = cv2.FONT_HERSHEY_SIMPLEX, (0,0,0), size_px + 2*border_px
    id_txt = f"ID: {marker_id}"; (tw,th),_ = cv2.getTextSize(id_txt,font,0.7,2)
    cv2.putText(img_bgr_bord,id_txt,(tot_sz-tw-max(5,int(border_px*0.1)),tot_sz-max(5,int(border_px*0.1))),font,0.7,txt_col,2)
    if border_px >= 20:
        ax_th, arr_l, tip_l = 2, max(10,int(border_px*0.6)), 0.3; cx,cy,cz = (0,0,255),(0,255,0),(255,0,0); sfs,sft,pad = 0.5,1,max(3,int(border_px*0.1))
        orig_x = (border_px, border_px + size_px + border_px//2); cv2.arrowedLine(img_bgr_bord, orig_x, (orig_x[0]+arr_l, orig_x[1]), cx, ax_th, tipLength=tip_l)
        (xlw,xlh),_ = cv2.getTextSize("X",font,sfs,sft); cv2.putText(img_bgr_bord,"X",(orig_x[0]+arr_l+pad, orig_x[1]+xlh//2),font,sfs,cx,sft)
        orig_y = (border_px - border_px//2, border_px + size_px); cv2.arrowedLine(img_bgr_bord, orig_y, (orig_y[0], orig_y[1]-arr_l), cy, ax_th, tipLength=tip_l)
        (ylw,ylh),_ = cv2.getTextSize("Y",font,sfs,sft); cv2.putText(img_bgr_bord,"Y",(orig_y[0]-ylw//2, orig_y[1]-arr_l-pad),font,sfs,cy,sft)
        z_cen,z_rad = (border_px//2,border_px//2),max(5,border_px//4); cv2.circle(img_bgr_bord,z_cen,z_rad,cz,ax_th); cv2.circle(img_bgr_bord,z_cen,max(1,z_rad//3),cz,-1)
        (zlw,zlh),_ = cv2.getTextSize("Z",font,sfs,sft); cv2.putText(img_bgr_bord,"Z",(z_cen[0]+z_rad+pad,z_cen[1]+zlh//2),font,sfs,cz,sft)
        (zoutw,zouth),_ = cv2.getTextSize("(out)",font,sfs*0.8,sft); cv2.putText(img_bgr_bord,"(out)",(z_cen[0]+z_rad+pad,z_cen[1]+zlh//2+zouth+2),font,sfs*0.8,cz,sft)
    if marker_id == REFERENCE_MARKER_ID and USE_REFERENCE_MARKER:
        ref_txt = "REFERENCE MARKER"; (trw,trh),_ = cv2.getTextSize(ref_txt,font,0.8,2); cv2.putText(img_bgr_bord,ref_txt,((tot_sz-trw)//2,border_px//2+trh//2 if border_px>trh else trh+5),font,0.8,txt_col,2)
    if save:
        fn_sfx = f"_ID{marker_id}" + ('_REFERENCE' if marker_id==REFERENCE_MARKER_ID and USE_REFERENCE_MARKER else '')
        fn = os.path.join(DATA_OUTPUT_DIRECTORY, PATTERNS_DIRECTORY_NAME, f"aruco_marker{fn_sfx}.png")
        try: cv2.imwrite(fn,img_bgr_bord); msg = f"Saved: {fn}"
        except Exception as e: msg = f"Save Error: {e}"
        if parent_tk_window: messagebox.showinfo("Marker Gen", msg, parent=parent_tk_window)
        else: print(msg)

def create_chessboard(squares_x, squares_y, square_size_px=100, save=True, parent_tk_window=None):
    pc,pr = squares_x+1,squares_y+1; w,h = pc*square_size_px,pr*square_size_px
    board = np.full((h,w),255,dtype=np.uint8)
    for r_idx in range(pr):
        for c_idx in range(pc):
            if (c_idx+r_idx)%2==0: board[r_idx*square_size_px:(r_idx+1)*square_size_px, c_idx*square_size_px:(c_idx+1)*square_size_px]=0
    instr_h=100; instr = np.full((instr_h,w),255,dtype=np.uint8)
    txt1 = "Camera Calibration Pattern"; (tw1,th1),_ = cv2.getTextSize(txt1,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
    cv2.putText(instr,txt1,((w-tw1)//2,int(instr_h*0.4)),cv2.FONT_HERSHEY_SIMPLEX,0.8,0,2)
    txt2 = f"{squares_x}x{squares_y} internal corners, Square: {CHESSBOARD_SQUARE_SIZE_MM}mm"; (tw2,th2),_ = cv2.getTextSize(txt2,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.putText(instr,txt2,((w-tw2)//2,int(instr_h*0.8)),cv2.FONT_HERSHEY_SIMPLEX,0.6,0,2)
    final_img = np.vstack((instr,board))
    if save:
        fn = os.path.join(DATA_OUTPUT_DIRECTORY, PATTERNS_DIRECTORY_NAME, f"chessboard_{squares_x}x{squares_y}_corners_{CHESSBOARD_SQUARE_SIZE_MM}mm.png")
        try: cv2.imwrite(fn,final_img); msg = f"Saved: {fn}"
        except Exception as e: msg = f"Save Error: {e}"
        if parent_tk_window: messagebox.showinfo("Chess Gen", msg, parent=parent_tk_window)
        else: print(msg)

# %% ===========================================================================
# GUI CLASSES
# =============================================================================
class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tooltip = widget, text, None
        self.widget.bind("<Enter>", self.show_tooltip); self.widget.bind("<Leave>", self.hide_tooltip)
    def show_tooltip(self, event):
        x, y, _, _ = self.widget.bbox("insert"); x += self.widget.winfo_rootx() + 0; y += self.widget.winfo_rooty() - 40
        self.tooltip = tk.Toplevel(self.widget); self.tooltip.wm_overrideredirect(True); self.tooltip.wm_geometry(f"+{x}+{y}")
        bg = "#2b2b2b" if sv_ttk.get_theme() == "dark" else "#ffffe0"; fg = "#ffffff" if sv_ttk.get_theme() == "dark" else "#000000"
        label = ttk.Label(self.tooltip, text=self.text, justify='left', background=bg, foreground=fg, relief='solid', borderwidth=1, padding=4)
        label.pack(ipadx=1)
    def hide_tooltip(self, event):
        if self.tooltip: self.tooltip.destroy()
        self.tooltip = None

class CustomInputDialog(tk.Toplevel):
    def __init__(self, parent, title, fields):
        super().__init__(parent); self.transient(parent); self.grab_set(); self.title(title)
        self.result, self.fields, self.vars = None, fields, {}
        body = ttk.Frame(self, padding="10 10 10 10")
        self.initial_focus = self.create_body(body); body.pack(padx=5, pady=5)
        self.create_buttons()
        if self.initial_focus: self.initial_focus.focus_set()
        self.protocol("WM_DELETE_WINDOW", self.cancel); self.geometry(f"+{parent.winfo_rootx()+50}+{parent.winfo_rooty()+50}"); self.wait_window(self)
    def create_body(self, master):
        initial_focus = None
        for i, field in enumerate(self.fields):
            label_text, var_type, initial_value, options = field.get("label", ""), field.get("type", "str"), field.get("initial", ""), field.get("options", [])
            ttk.Label(master, text=label_text).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            if var_type == "int": self.vars[label_text] = tk.IntVar(value=initial_value)
            elif var_type == "float": self.vars[label_text] = tk.DoubleVar(value=initial_value)
            else: self.vars[label_text] = tk.StringVar(value=initial_value)
            widget = ttk.Combobox(master, textvariable=self.vars[label_text], values=options, state="readonly") if options else ttk.Entry(master, textvariable=self.vars[label_text])
            widget.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
            if not initial_focus: initial_focus = widget
        master.columnconfigure(1, weight=1)
        return initial_focus
    def create_buttons(self):
        box = ttk.Frame(self)
        ok_button = ttk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE); ok_button.pack(side=tk.LEFT, padx=5, pady=5)
        cancel_button = ttk.Button(box, text="Cancel", width=10, command=self.cancel); cancel_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.bind("<Return>", self.ok); self.bind("<Escape>", self.cancel); box.pack()
    def ok(self, event=None):
        if not self.validate(): self.initial_focus.focus_set(); return
        self.withdraw(); self.update_idletasks(); self.apply(); self.cancel()
    def cancel(self, event=None): self.master.focus_set(); self.destroy()
    def validate(self):
        for field in self.fields:
            label_text, var, rules = field.get("label"), self.vars[field.get("label")], field.get("validation", {})
            try:
                val = var.get()
                if "min" in rules and val < rules["min"]: messagebox.showwarning("Validation Error", f"{label_text} must be >= {rules['min']}.", parent=self); return False
                if "max" in rules and val > rules["max"]: messagebox.showwarning("Validation Error", f"{label_text} must be <= {rules['max']}.", parent=self); return False
            except tk.TclError: messagebox.showwarning("Validation Error", f"Invalid value for {label_text}.", parent=self); return False
        return True
    def apply(self): self.result = {label: var.get() for label, var in self.vars.items()}

class OfflineProcessingGUI(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master); self.title("Offline Video Processor"); self.geometry("700x450"); self.transient(master); self.grab_set()
        self.temp_led_roi, self.temp_led_baseline = None, None
        ttk.Label(self, text="Select a video file for offline ArUco processing.").pack(pady=(10,5))
        file_frame = ttk.Frame(self, padding="10 5"); file_frame.pack(pady=5, padx=10, fill=tk.X)
        self.filepath_var = tk.StringVar(); ttk.Entry(file_frame, textvariable=self.filepath_var, width=50, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file); self.browse_button.pack(side=tk.LEFT); ToolTip(self.browse_button, "Browse for a video file.")
        led_frame = ttk.LabelFrame(self, text="LED Synchronization", padding="10"); led_frame.pack(pady=5, padx=10, fill=tk.X)
        self.led_roi_button = ttk.Button(led_frame, text="Define LED Area for this Video", command=self.define_led_roi_for_video, state=tk.DISABLED); self.led_roi_button.pack(pady=5, fill=tk.X)
        ToolTip(self.led_roi_button, "Opens the first frame of the video to define the LED's location and 'off' state.\nRequired if LED detection is enabled in Settings.")
        self.led_status_label = ttk.Label(led_frame, text="LED area not defined for this video."); self.led_status_label.pack(pady=(0, 5))
        options_frame = ttk.Frame(self, padding="10 0"); options_frame.pack(pady=5, padx=10, fill=tk.X)
        self.ref_marker_var = tk.BooleanVar(value=USE_REFERENCE_MARKER); self.ref_marker_check = ttk.Checkbutton(options_frame, text=f"Use Reference Marker (ID: {REFERENCE_MARKER_ID})", variable=self.ref_marker_var); self.ref_marker_check.pack(pady=5, anchor='w'); ToolTip(self.ref_marker_check, "Calculate poses relative to the reference marker.")
        time_input_frame = ttk.LabelFrame(self, text="Optional: Video Start Timestamp", padding="10"); time_input_frame.pack(pady=5, padx=10, fill=tk.X)
        self.video_start_date_var = tk.StringVar(); self.video_start_time_var = tk.StringVar()
        ttk.Label(time_input_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2); ttk.Entry(time_input_frame, textvariable=self.video_start_date_var, width=15).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(time_input_frame, text="Start Time (HH:MM:SS):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2); ttk.Entry(time_input_frame, textvariable=self.video_start_time_var, width=15).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        time_input_frame.columnconfigure(1, weight=1); ToolTip(time_input_frame, "If provided, timestamps in the log file will be absolute.")
        self.process_button = ttk.Button(self, text="Start Processing", command=self.start_processing, state=tk.DISABLED, style="Accent.TButton"); self.process_button.pack(pady=15, ipady=5); ToolTip(self.process_button, "Begin processing.")
        self.status_label = ttk.Label(self, text="Status: Idle", anchor='w'); self.status_label.pack(pady=5, fill=tk.X, padx=10)
        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=100, mode='determinate'); self.progress_bar.pack(pady=(5,10), fill=tk.X, padx=10)
    def browse_file(self):
        filename = filedialog.askopenfilename(title="Select Video", filetypes=(("Video files", "*.mov *.mp4 *.avi *.mkv"),("All files", "*.*")), parent=self)
        if filename:
            self.filepath_var.set(filename); self.process_button.config(state=tk.NORMAL); self.led_roi_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Selected: {os.path.basename(filename)}"); self.progress_bar["value"] = 0
            self.temp_led_roi, self.temp_led_baseline = None, None
            self.led_status_label.config(text="LED area not defined for this video.")
    def define_led_roi_for_video(self):
        video_path = self.filepath_var.get()
        if not video_path: messagebox.showerror("Error", "Please select a video file first.", parent=self); return
        cap = cv2.VideoCapture(video_path); ret, frame = cap.read(); cap.release()
        if not ret: messagebox.showerror("Video Error", "Could not read the first frame of the video.", parent=self); return
        messagebox.showinfo("Define LED Area", "The first frame of your video will be shown.\n\n1. IMPORTANT: The LED should be OFF in this frame.\n2. Click and drag to draw a box around the LED.\n3. Press ENTER or SPACE to confirm.\n4. Press 'c' or ESC to cancel.", parent=self)
        window_name = "Select LED Area (on first frame) - Press ENTER to Confirm"; cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True); cv2.destroyWindow(window_name)
        if roi and roi[2] > 0 and roi[3] > 0:
            self.temp_led_roi = roi
            x, y, w, h = [int(v) for v in roi]
            roi_patch = frame[y:y+h, x:x+w]
            self.temp_led_baseline = np.mean(roi_patch[:, :, 2])
            self.led_status_label.config(text=f"LED area SET for this video. Baseline: {self.temp_led_baseline:.2f}")
        else: self.led_status_label.config(text="LED area selection cancelled.")
    def start_processing(self):
        video_path = self.filepath_var.get()
        if not video_path: messagebox.showerror("Error", "No video selected.", parent=self); return
        if ENABLE_LED_DETECTION and self.temp_led_roi is None and not messagebox.askyesno("Warning", "LED Detection is enabled, but no area has been defined for this video.\n\nContinue without LED detection?", icon='warning', parent=self): return
        use_ref_for_this_run, video_start_date, video_start_time = self.ref_marker_var.get(), self.video_start_date_var.get(), self.video_start_time_var.get()
        self.process_button.config(state=tk.DISABLED); self.browse_button.config(state=tk.DISABLED); self.ref_marker_check.config(state=tk.DISABLED); self.led_roi_button.config(state=tk.DISABLED)
        for child in self.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Entry): widget.config(state=tk.DISABLED)
        self.status_label.config(text="Processing..."); self.progress_bar["value"] = 0; self.update_idletasks()
        try:
            process_video_offline(video_path, use_ref_for_this_run, self.status_label, self.progress_bar, video_start_date, video_start_time, self.temp_led_roi, self.temp_led_baseline)
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error: {e}", parent=self); self.status_label.config(text=f"Error: {e}")
        finally:
            self.process_button.config(state=tk.NORMAL if self.filepath_var.get() else tk.DISABLED); self.browse_button.config(state=tk.NORMAL); self.ref_marker_check.config(state=tk.NORMAL); self.led_roi_button.config(state=tk.NORMAL)
            for child in self.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for widget in child.winfo_children():
                        if isinstance(widget, ttk.Entry): widget.config(state=tk.NORMAL)

class SettingsDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master); self.title("Application Settings"); self.transient(master); self.grab_set(); self.initial_ref_marker_id = REFERENCE_MARKER_ID
        self.use_ref_marker_var, self.ref_marker_id_var, self.record_video_var, self.reverse_display_var = tk.BooleanVar(value=USE_REFERENCE_MARKER), tk.IntVar(value=REFERENCE_MARKER_ID), tk.BooleanVar(value=RECORD_VIDEO), tk.BooleanVar(value=REVERSE_CAMERA_DISPLAY)
        self.log_all_marker_data_var, self.camera_index_var, self.marker_size_var = tk.BooleanVar(value=LOG_ALL_MARKER_DATA), tk.IntVar(value=CAMERA_INDEX), tk.DoubleVar(value=MARKER_SIZE_METERS)
        self.cam_res_width_var, self.cam_res_height_var, self.show_plots_var, self.enable_led_detection_var = tk.IntVar(value=CAMERA_RESOLUTION[0]), tk.IntVar(value=CAMERA_RESOLUTION[1]), tk.BooleanVar(value=SHOW_REALTIME_PLOTS), tk.BooleanVar(value=ENABLE_LED_DETECTION)
        main_frame = ttk.Frame(self, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        vis_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10"); vis_frame.pack(fill=tk.X, pady=(0, 10))
        c1 = ttk.Checkbutton(vis_frame, variable=self.show_plots_var, text="Enable real-time plots panel"); c1.pack(anchor='w', pady=2); ToolTip(c1, "Show/hide live data plots.")
        c2 = ttk.Checkbutton(vis_frame, variable=self.reverse_display_var, text="Reverse live camera display"); c2.pack(anchor='w', pady=2); ToolTip(c2, "Flips the live camera view horizontally.")
        log_frame = ttk.LabelFrame(main_frame, text="Tracking & Logging", padding="10"); log_frame.pack(fill=tk.X, pady=5)
        c3 = ttk.Checkbutton(log_frame, variable=self.use_ref_marker_var, text="Enable Reference Marker by default", command=self._toggle_ref_id_entry); c3.pack(anchor='w', pady=2); ToolTip(c3, "Report poses relative to a static world-frame marker.")
        ref_id_frame = ttk.Frame(log_frame); ref_id_frame.pack(fill=tk.X, padx=20); ttk.Label(ref_id_frame, text="Reference Marker ID:").pack(side=tk.LEFT)
        self.ref_id_entry = ttk.Entry(ref_id_frame, textvariable=self.ref_marker_id_var, width=7); self.ref_id_entry.pack(side=tk.LEFT); ToolTip(self.ref_id_entry, "The ID of the static reference marker.")
        c4 = ttk.Checkbutton(log_frame, variable=self.record_video_var, text="Record video by default (Live mode)"); c4.pack(anchor='w', pady=2); ToolTip(c4, "Start recording automatically in live sessions.")
        c5 = ttk.Checkbutton(log_frame, variable=self.log_all_marker_data_var, text="Log all individual marker data"); c5.pack(anchor='w', pady=2); ToolTip(c5, "Creates an extra CSV log for every detected marker.")
        c6 = ttk.Checkbutton(log_frame, variable=self.enable_led_detection_var, text="Enable Red LED sync detection"); c6.pack(anchor='w', pady=2); ToolTip(c6, "Enables monitoring a red LED for synchronization.\nDefine the area from the Tools menu.")
        cam_frame = ttk.LabelFrame(main_frame, text="Hardware Parameters", padding="10"); cam_frame.pack(fill=tk.X, pady=5)
        g = ttk.Frame(cam_frame); g.pack(fill=tk.X)
        ttk.Label(g, text="Camera Index:").grid(row=0, column=0, sticky='w', pady=2); e1 = ttk.Entry(g, textvariable=self.camera_index_var, width=10); e1.grid(row=0, column=1, sticky='w'); ToolTip(e1, "System index of the camera (e.g., 0, 1).")
        ttk.Label(g, text="ArUco Marker Size (m):").grid(row=1, column=0, sticky='w', pady=2); e2 = ttk.Entry(g, textvariable=self.marker_size_var, width=10); e2.grid(row=1, column=1, sticky='w'); ToolTip(e2, "Physical side length of the square markers, in meters.")
        ttk.Label(g, text="Camera Resolution:").grid(row=2, column=0, sticky='w', pady=2)
        res_frame = ttk.Frame(g); res_frame.grid(row=2, column=1, sticky='w'); e3 = ttk.Entry(res_frame, textvariable=self.cam_res_width_var, width=7); e3.pack(side=tk.LEFT); ttk.Label(res_frame, text=" x ").pack(side=tk.LEFT); e4 = ttk.Entry(res_frame, textvariable=self.cam_res_height_var, width=7); e4.pack(side=tk.LEFT); ToolTip(res_frame, "Desired camera capture resolution (width x height).")
        btn_fr = ttk.Frame(main_frame); btn_fr.pack(pady=(15,5)); ttk.Button(btn_fr, text="Apply & Save", command=self.apply_settings, style="Accent.TButton").pack(side=tk.LEFT, padx=5); ttk.Button(btn_fr, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        self._toggle_ref_id_entry()
    def _toggle_ref_id_entry(self): self.ref_id_entry.config(state=tk.NORMAL if self.use_ref_marker_var.get() else tk.DISABLED)
    def apply_settings(self):
        global USE_REFERENCE_MARKER,REFERENCE_MARKER_ID,RECORD_VIDEO,REVERSE_CAMERA_DISPLAY,CAMERA_INDEX,MARKER_SIZE_METERS,CAMERA_RESOLUTION,LOG_ALL_MARKER_DATA,SHOW_REALTIME_PLOTS,ENABLE_LED_DETECTION
        v_ref_id = self.initial_ref_marker_id
        if self.use_ref_marker_var.get():
            try:
                tid = self.ref_marker_id_var.get()
                if tid < 0: messagebox.showwarning("Invalid ID","Ref ID must be non-negative.",parent=self); self.ref_marker_id_var.set(self.initial_ref_marker_id); return
                v_ref_id = tid
            except tk.TclError: messagebox.showwarning("Invalid ID","Ref ID must be an integer.",parent=self); self.ref_marker_id_var.set(self.initial_ref_marker_id); return
        try:
            n_cam_idx, n_marker_size, n_cam_w, n_cam_h = self.camera_index_var.get(), self.marker_size_var.get(), self.cam_res_width_var.get(), self.cam_res_height_var.get()
            if n_cam_idx < 0: raise ValueError("Camera index must be >= 0.")
            if n_marker_size <= 0: raise ValueError("Marker size must be > 0.")
            if n_cam_w <= 0 or n_cam_h <= 0: raise ValueError("Resolution must be positive.")
        except (tk.TclError, ValueError) as e: messagebox.showwarning("Invalid Input", f"Error: {e}", parent=self); return
        SHOW_REALTIME_PLOTS, USE_REFERENCE_MARKER, REFERENCE_MARKER_ID = self.show_plots_var.get(), self.use_ref_marker_var.get(), v_ref_id
        RECORD_VIDEO, REVERSE_CAMERA_DISPLAY, LOG_ALL_MARKER_DATA = self.record_video_var.get(), self.reverse_display_var.get(), self.log_all_marker_data_var.get()
        CAMERA_INDEX, MARKER_SIZE_METERS, CAMERA_RESOLUTION = n_cam_idx, n_marker_size, (n_cam_w, n_cam_h)
        ENABLE_LED_DETECTION = self.enable_led_detection_var.get()
        print(f"Settings updated: Ref={USE_REFERENCE_MARKER},ID={REFERENCE_MARKER_ID},Rec={RECORD_VIDEO},Rev={REVERSE_CAMERA_DISPLAY},LogAll={LOG_ALL_MARKER_DATA},CamIdx={CAMERA_INDEX},MarkerSize={MARKER_SIZE_METERS},CamRes={CAMERA_RESOLUTION},LED_Sync={ENABLE_LED_DETECTION}")
        save_settings()
        initialize_buoy_configs(); messagebox.showinfo("Settings Applied","Settings updated and saved.",parent=self); self.destroy()

class BuoyTrackerApp:
    def __init__(self, master):
        self.master = master; master.title("ArUco-6DOF Buoy Tracker"); master.geometry("500x550")
        if THEME_AVAILABLE: sv_ttk.set_theme("light")
        try: master.iconphoto(True, tk.PhotoImage(file=APP_ICON_PATH))
        except tk.TclError: print(f"Warning: Icon '{APP_ICON_PATH}' not found.")
        master.columnconfigure(0, weight=1); master.rowconfigure(0, weight=1); self._create_menubar()
        main_frame = ttk.Frame(master, padding="15"); main_frame.grid(row=0, column=0, sticky="nsew"); main_frame.columnconfigure(0, weight=1)
        ttk.Label(main_frame, text="Buoy Tracking & Analysis Tool", font=("", 16, "bold")).pack(pady=(0, 20))
        core_frame = ttk.LabelFrame(main_frame, text="Core Functions", padding="10"); core_frame.pack(fill=tk.X, pady=10); core_frame.columnconfigure(0, weight=1)
        live_btn = ttk.Button(core_frame, text="Start Live Tracking Session", command=self.ui_live_tracking, style="Accent.TButton"); live_btn.pack(fill=tk.X, pady=5, ipady=8); ToolTip(live_btn, "Begin real-time tracking from a camera.")
        offline_btn = ttk.Button(core_frame, text="Process Recorded Video", command=self.ui_launch_offline_processor); offline_btn.pack(fill=tk.X, pady=5, ipady=4); ToolTip(offline_btn, "Analyze a pre-recorded video.")
        calib_frame = ttk.LabelFrame(main_frame, text="Calibration & Setup Tools", padding="10"); calib_frame.pack(fill=tk.X, pady=10); calib_frame.columnconfigure(0, weight=1); calib_frame.columnconfigure(1, weight=1)
        cam_calib_btn = ttk.Button(calib_frame, text="Calibrate Camera", command=self.ui_calibrate_camera); cam_calib_btn.grid(row=0, column=0, sticky="ew", padx=5, pady=5); ToolTip(cam_calib_btn, "Calibrate camera intrinsics with a chessboard.")
        buoy_calib_btn = ttk.Button(calib_frame, text="Calibrate Buoy Geometry", command=self.ui_calibrate_buoy_geometry); buoy_calib_btn.grid(row=0, column=1, sticky="ew", padx=5, pady=5); ToolTip(buoy_calib_btn, "Calibrate relative marker poses on a buoy.")
        led_roi_btn = ttk.Button(calib_frame, text="Define LED Area (Live)", command=self.ui_define_led_roi); led_roi_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(10,5)); ToolTip(led_roi_btn, "Define the screen area to monitor for the sync LED.\nDo this while the LED is OFF.")
        self.status_var = tk.StringVar(value="Ready. Welcome!"); status_bar = ttk.Frame(master, style="Card.TFrame", padding=5); status_bar.grid(row=1, column=0, sticky="ew"); ttk.Label(status_bar, textvariable=self.status_var).pack(anchor='w')
    def _create_menubar(self):
        menubar = tk.Menu(self.master); self.master.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0); menubar.add_cascade(label="File", menu=file_menu); file_menu.add_command(label="Settings", command=self.ui_configure_settings); file_menu.add_separator(); file_menu.add_command(label="Exit", command=self.master.quit)
        tools_menu = tk.Menu(menubar, tearoff=0); menubar.add_cascade(label="Tools", menu=tools_menu)
        gen_menu = tk.Menu(tools_menu, tearoff=0); tools_menu.add_cascade(label="Generate Pattern", menu=gen_menu); gen_menu.add_command(label="Single ArUco Marker...", command=self.ui_generate_marker); gen_menu.add_command(label="Reference Marker...", command=self.ui_generate_reference_marker); gen_menu.add_command(label="Chessboard...", command=self.ui_generate_chessboard)
        tools_menu.add_separator(); tools_menu.add_command(label="Define LED Area (Live)...", command=self.ui_define_led_roi)
        help_menu = tk.Menu(menubar, tearoff=0); menubar.add_cascade(label="Help", menu=help_menu)
        if THEME_AVAILABLE: help_menu.add_command(label="Toggle Dark/Light Mode", command=sv_ttk.toggle_theme)
        help_menu.add_command(label="About", command=self.ui_show_about)
    def ui_show_about(self): messagebox.showinfo("About Buoy Tracker", f"ArUco-Based 6-DOF Buoy Tracker\nVersion: 1.3\nAuthor: {__doc__.split('Author: ')[1].splitlines()[0]}\nContact: {__doc__.split('Contact: ')[1].splitlines()[0]}", parent=self.master)
    def ui_generate_marker(self):
        fields = [{"label": "Marker ID", "type": "int", "initial": 0, "validation": {"min": 0, "max": 249}}, {"label": "Image Size (px)", "type": "int", "initial": 600}, {"label": "Border Width (px)", "type": "int", "initial": 50}]
        dialog = CustomInputDialog(self.master, "Generate ArUco Marker", fields)
        if dialog.result:
            mid, spx, bpx = dialog.result.values(); self.status_var.set(f"Generating marker ID {mid}..."); self.master.update_idletasks()
            img = generate_aruco_marker(mid, spx, bpx, True, self.master)
            if img is not None: cv2.imshow(f"Generated ArUco - ID:{mid}", img); cv2.waitKey(0); cv2.destroyWindow(f"Generated ArUco - ID:{mid}")
            self.status_var.set("Ready.")
    def ui_generate_reference_marker(self):
        fields = [{"label": "Image Size (px)", "type": "int", "initial": 600}, {"label": "Border Width (px)", "type": "int", "initial": 50}]
        dialog = CustomInputDialog(self.master, f"Generate Reference Marker (ID: {REFERENCE_MARKER_ID})", fields)
        if dialog.result:
            spx, bpx = dialog.result.values(); self.status_var.set(f"Generating ref marker ID {REFERENCE_MARKER_ID}..."); self.master.update_idletasks()
            img = generate_aruco_marker(REFERENCE_MARKER_ID, spx, bpx, True, self.master)
            if img is not None: cv2.imshow(f"Generated Reference - ID:{REFERENCE_MARKER_ID}", img); cv2.waitKey(0); cv2.destroyWindow(f"Generated Reference - ID:{REFERENCE_MARKER_ID}")
            self.status_var.set("Ready.")
    def ui_generate_chessboard(self):
        sx, sy = CALIBRATION_CHESSBOARD_SHAPE; fields = [{"label": "Square Size (px)", "type": "int", "initial": 100}]
        dialog = CustomInputDialog(self.master, f"Generate {sx}x{sy} Chessboard", fields)
        if dialog.result:
            spx = list(dialog.result.values())[0]; self.status_var.set("Generating chessboard..."); self.master.update_idletasks()
            img = create_chessboard(sx, sy, spx, True, self.master)
            if img is not None: cv2.imshow(f"Generated Chessboard ({sx}x{sy})", img); cv2.waitKey(0); cv2.destroyWindow(f"Generated Chessboard ({sx}x{sy})")
            self.status_var.set("Ready.")
    def ui_calibrate_camera(self):
        self.status_var.set("Starting camera calibration..."); self.master.update_idletasks()
        messagebox.showinfo("Camera Calibration", "Camera view will open.\n- Aim at the chessboard.\n- Press 'c' to capture.\n- Press 'q' to finish & calibrate.", parent=self.master)
        calibrate_camera(self.master); self.status_var.set("Ready.")
    def ui_define_led_roi(self):
        global LED_DETECTION_ROI, LED_OFF_BASELINE_BRIGHTNESS
        if not ENABLE_LED_DETECTION: messagebox.showwarning("Feature Disabled", "Enable LED detection in File -> Settings first.", parent=self.master); return
        messagebox.showinfo("Define LED Area (Live)", "Camera view will open.\n\n1. Position camera so the RED LED is visible and OFF.\n2. Press 's' to freeze the frame.\n3. Click and drag a box around the LED.\n4. Press ENTER to confirm, or ESC to cancel.", parent=self.master)
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
        if not cap.isOpened(): messagebox.showerror("Camera Error", f"Cannot open camera {CAMERA_INDEX}.", parent=self.master); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
        window_name, frozen_frame = "Live ROI Selection - Press 's' to freeze and select", None
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret: messagebox.showerror("Camera Error", "Failed to grab a frame.", parent=self.master); break
            display_frame = frame.copy(); h, w, _ = display_frame.shape
            cv2.putText(display_frame, "Press 's' to freeze frame for selection", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 255, 255), OVERLAY_THICKNESS)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): frozen_frame = frame.copy(); break
            if key in [ord('c'), 27]: break
        cap.release()
        if frozen_frame is not None:
            roi_window_name = "Draw Box, Press ENTER to Confirm"
            cv2.namedWindow(roi_window_name, cv2.WINDOW_NORMAL)
            roi = cv2.selectROI(roi_window_name, frozen_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(roi_window_name)
            if roi and roi[2] > 0 and roi[3] > 0:
                LED_DETECTION_ROI = roi; x, y, w, h = [int(v) for v in roi]
                roi_patch = frozen_frame[y:y+h, x:x+w]
                LED_OFF_BASELINE_BRIGHTNESS = np.mean(roi_patch[:, :, 2])
                save_settings()
                self.status_var.set(f"Global LED ROI set. Baseline: {LED_OFF_BASELINE_BRIGHTNESS:.2f}")
                messagebox.showinfo("Success", f"Global LED detection area set and saved.\n'OFF' baseline: {LED_OFF_BASELINE_BRIGHTNESS:.2f}", parent=self.master)
            else: self.status_var.set("Global LED ROI selection cancelled.")
        else: self.status_var.set("Global LED ROI selection cancelled.")
        cv2.destroyAllWindows()
    def ui_live_tracking(self):
        self.status_var.set("Preparing live session..."); self.master.update_idletasks()
        log_all_text = "Enabled" if LOG_ALL_MARKER_DATA else "Disabled"
        led_sync_text = "Enabled" if ENABLE_LED_DETECTION else "Disabled"
        if ENABLE_LED_DETECTION and LED_DETECTION_ROI is None: led_sync_text += " (WARNING: Area not defined!)"
        conf_msg = f"Start live tracking with these settings?\n\n  • Camera: {CAMERA_INDEX} @ {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]}\n  • Reference Frame: {'Enabled (ID ' + str(REFERENCE_MARKER_ID) + ')' if USE_REFERENCE_MARKER else 'Disabled'}\n  • LED Sync: {led_sync_text}\n  • Auto-record: {'Enabled' if RECORD_VIDEO else 'Disabled'}"
        if messagebox.askyesno("Confirm Live Session", conf_msg, parent=self.master):
            self.status_var.set("Live tracking active... (See OpenCV window)")
            self.master.update_idletasks()
            try: track_buoy_with_aruco_6dof(self.master)
            except Exception as e: messagebox.showerror("Live Tracking Error", f"Error: {e}", parent=self.master); self.status_var.set(f"Live tracking error: {e}")
            self.status_var.set("Ready.")
        else: self.status_var.set("Live tracking cancelled.")
    def ui_launch_offline_processor(self):
        self.status_var.set("Opening Offline Processor..."); self.master.update_idletasks()
        opg = OfflineProcessingGUI(self.master); opg.protocol("WM_DELETE_WINDOW", lambda: [self.status_var.set("Ready."), opg.destroy()])
    def ui_configure_settings(self):
        self.status_var.set("Opening settings..."); self.master.update_idletasks()
        sd = SettingsDialog(self.master); self.master.wait_window(sd); self.status_var.set("Settings window closed.")
    def ui_calibrate_buoy_geometry(self):
        buoy_ids = list(BUOY_ID_TO_MARKER_IDS_MAP.keys())
        if not buoy_ids: messagebox.showerror("Error", "No buoys defined in configuration.", parent=self.master); return
        num_other_markers = len(BUOY_ID_TO_MARKER_IDS_MAP.get(buoy_ids[0], [0,0])) - 1
        num_frames_suggestion = BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR * max(1, num_other_markers) * 2
        fields = [{"label": "Buoy ID to Calibrate", "type": "int", "initial": buoy_ids[0], "options": buoy_ids}, {"label": "Total Views to Capture", "type": "int", "initial": max(10, num_frames_suggestion)}]
        dialog = CustomInputDialog(self.master, "Calibrate Buoy Geometry", fields)
        if dialog.result:
            buoy_id_to_calibrate, num_frames = dialog.result.values()
            self.status_var.set(f"Starting geometry calibration for Buoy ID: {buoy_id_to_calibrate}..."); self.master.update_idletasks()
            calibrate_buoy_marker_geometry(buoy_id_to_calibrate, self.master, num_frames_to_capture=num_frames)
            self.status_var.set(f"Buoy {buoy_id_to_calibrate} geometry calibration finished.")

# %% ===========================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    # Ensure all required directories exist
    os.makedirs(DATA_OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME), exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME), exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, PATTERNS_DIRECTORY_NAME), exist_ok=True)
    
    load_settings()  # Load settings from JSON file on startup
    
    print("Application starting...")
    root = tk.Tk()
    app = BuoyTrackerApp(root)
    root.mainloop()
    print("Application closed.")
