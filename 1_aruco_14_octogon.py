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

try:
    import sv_ttk
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False
    print("Warning: 'sv-ttk' library not found. GUI will use the default theme.")
    print("For a modern look, run: pip install sv-ttk")

import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

## SECTION: GLOBAL CONFIGURATION CONSTANTS
# Camera settings
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (3840, 2160)
CAMERA_FPS = 30
REVERSE_CAMERA_DISPLAY = True

# Display and Plotting settings
SHOW_REALTIME_PLOTS = True
PLOT_WIDTH_PX = 1500
INFO_PANEL_WIDTH_PX = 450
OVERLAY_FONT_SCALE = 1.2
OVERLAY_THICKNESS = 2
AXES_THICKNESS = 6
TRAJECTORY_THICKNESS = 4
PLOT_HISTORY_LENGTH = 150 # Number of data points to display in real-time plots
PLOT_TITLE_FONTSIZE = 32
PLOT_LABEL_FONTSIZE = 32
PLOT_TICK_FONTSIZE = 32
PLOT_LINE_WIDTH = 2.0
PLOT_LEGEND_FONTSIZE = 'medium'

# Video recording settings
RECORD_VIDEO = False
VIDEO_CODEC = 'mp4v' # H.264 compatible codec. Try 'XVID' if 'mp4v' fails on Windows.
VIDEO_FPS = 30
VIDEO_EXTENSION = '.mp4'

# ArUco marker settings
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
MARKER_SIZE_METERS = 0.035#0.1577 # Physical side length of the square markers

# Reference Marker settings
REFERENCE_MARKER_ID = 0
USE_REFERENCE_MARKER = True
ENABLE_REFERENCE_FILTER = True # Apply a low-pass filter to the reference marker pose
REF_FILTER_ALPHA_T = 0.3 # Alpha for translation filter (0=heavy filter, 1=no filter)
REF_FILTER_ALPHA_R = 0.3 # Alpha for rotation filter

# LED Synchronization settings
ENABLE_LED_DETECTION = False
LED_DETECTION_ROI = None # (x, y, width, height) of the ROI for LED detection
LED_OFF_BASELINE_BRIGHTNESS = 0.0 # Mean red channel brightness when LED is OFF
LED_BRIGHTNESS_THRESHOLD = 30.0 # Threshold above baseline to consider LED ON

# Buoy configuration (mapping buoy IDs to marker IDs)
NUM_BUOYS = 10
MARKERS_PER_BUOY = 8 # Assumes 3 markers per buoy for triangular geometry
BUOY_ID_TO_MARKER_IDS_MAP = {
    i + 1: list(range(10 + i * 10, 10 + i * 10 + MARKERS_PER_BUOY))
    for i in range(NUM_BUOYS)
}
MARKER_ID_TO_BUOY_ID_MAP = {
    marker_id: buoy_id
    for buoy_id, marker_ids_list in BUOY_ID_TO_MARKER_IDS_MAP.items()
    for marker_id in marker_ids_list
}

# Ideal marker geometry relative to X0 (first marker of the buoy)
# This is a fallback/default if buoy-specific calibration is not found.
TRIANGLE_SIDE_LENGTH_METERS = 0.21
_s = TRIANGLE_SIDE_LENGTH_METERS
MARKER_RELATIVE_GEOMETRY = {
    0: {'name': 'X0', 'rel_tvec_to_X0': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'rel_rvec_to_X0': np.array([0.0, 0.0, 0.0], dtype=np.float32)},
    1: {'name': 'X1', 'rel_tvec_to_X0': np.array([_s, 0.0, 0.0], dtype=np.float32),
        'rel_rvec_to_X0': np.array([0.0, np.deg2rad(120), 0.0], dtype=np.float32)}, # X1 is at _s distance from X0, rotated by 120 degrees around Y axis (in X0 frame)
    2: {'name': 'X2', 'rel_tvec_to_X0': np.array([_s * np.cos(np.pi/3), 0.0, _s * np.sin(np.pi/3)], dtype=np.float32),
        'rel_rvec_to_X0': np.array([0.0, np.deg2rad(240), 0.0], dtype=np.float32)} # X2 at another vertex, rotated 240 degrees
}

# Buoy Center of Gravity (CoG) offset from the buoy's reference frame (BRF).
# This vector translates from BRF origin to the physical CoG of the buoy.
# BUOY_COG_OFFSET_METERS = 0#.37768 # HMM: manually set. For a different system, do we need this?
# BUOY_COG_OFFSET_VECTOR_ASSEMBLY_FRAME = np.array(
#     [0, -BUOY_COG_OFFSET_METERS, 0*-1/3 * TRIANGLE_SIDE_LENGTH_METERS], dtype=np.float32
# ) # HMM: added zero offset for office testing

# Alternatively, the OCTOGON
# Octagon marker geometry relative to X0 (first marker of the buoy)
# Face-to-opposite-face distance is 9.1 cm.
OCTOGON_FACE_TO_FACE_METERS = 0.091
OCTOGON_CENTER_TO_FACE_METERS = OCTOGON_FACE_TO_FACE_METERS / 2
OCTOGON_ANGLE_STEP_RAD = 2 * np.pi / MARKERS_PER_BUOY

_r8 = OCTOGON_CENTER_TO_FACE_METERS

MARKER_RELATIVE_GEOMETRY = {
    i: {
        "name": f"X{i}",
        "rel_tvec_to_X0": np.array(
            [
                _r8 * np.sin(i * OCTOGON_ANGLE_STEP_RAD),
                0.0,
                _r8 * (np.cos(i * OCTOGON_ANGLE_STEP_RAD) - 1.0),
            ],
            dtype=np.float32,
        ),
        "rel_rvec_to_X0": np.array(
            [
                0.0,
                i * OCTOGON_ANGLE_STEP_RAD,
                0.0,
            ],
            dtype=np.float32,
        ),
    }
    for i in range(MARKERS_PER_BUOY)
}

# Buoy Center of Gravity (CoG) offset from the buoy's reference frame (BRF).
# X0 is on one face. The center is half the face-to-face distance behind X0.
BUOY_COG_OFFSET_METERS = OCTOGON_CENTER_TO_FACE_METERS
BUOY_COG_OFFSET_VECTOR_ASSEMBLY_FRAME = np.array(
    [0.0, 0.0, -BUOY_COG_OFFSET_METERS],
    dtype=np.float32,
)

# File and directory paths
DATA_OUTPUT_DIRECTORY = "buoy_tracking_data"
EXPERIMENTS_BASE_DIRECTORY_NAME = "experiments"
CONFIG_DIRECTORY_NAME = "config"
PATTERNS_DIRECTORY_NAME = "generated_patterns"

CALIBRATION_FILE = "camera_calibration.npz"
BUOY_GEOMETRY_CALIBRATION_FILE = "buoy_marker_calibrations.json"
SETTINGS_FILE = "app_settings.json"
APP_ICON_PATH = 'aruco_icon.png'

# Default camera calibration if no file is found
DEFAULT_CAMERA_MATRIX = np.array([
    [1000, 0, CAMERA_RESOLUTION[0]/2],
    [0, 1000, CAMERA_RESOLUTION[1]/2],
    [0, 0, 1]], dtype=np.float32)
DEFAULT_DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

# Calibration chessboard settings
CALIBRATION_CHESSBOARD_SHAPE = (9, 6) # Number of internal corners (cols, rows)
CHESSBOARD_SQUARE_SIZE_MM = 26.9 # Physical size of one square on the chessboard in mm
BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR = 5 # Min good samples required per (X0, Xi) pair for buoy calibration

# Logging and trajectory settings
LOG_ALL_MARKER_DATA = False # Log individual marker poses in addition to buoy CoG
TRAJECTORY_LENGTH = 30 # Number of past CoG positions to draw

# Overlay colors (BGR format)
TRAJECTORY_COLOR = (0, 0, 255) # Red
MARKER_INFO_COLOR = (0, 255, 0) # Green
BUOY_COG_INFO_COLOR = (255, 100, 0) # Orange-Blue
TIMESTAMP_COLOR = (0, 0, 255) # Red
REFERENCE_MARKER_COLOR = (255, 0, 255) # Magenta
RECORD_INDICATOR_COLOR = (0, 0, 255) # Red
OFFLINE_PROGRESS_COLOR = (255, 165, 0) # Orange

# Offline processing display
SHOW_OFFLINE_PROCESSING_PREVIEW = True
OFFLINE_PREVIEW_RESIZE_FACTOR = 0.8 # Factor to resize the preview window (to fit on smaller screens)

## END SECTION: GLOBAL CONFIGURATION CONSTANTS

## SECTION: HELPER CLASSES

class InfoPanelDrawer:
    """
    Draws an informational overlay panel on the video frame.
    Displays performance, system status, and buoy detection summaries.
    """
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

    def _reset_y_position(self):
        """Resets the Y-position for drawing text to the top of the panel."""
        self.y_pos = self.y_header_step

    def _draw_text_line(self, text, color, is_header=False, indent=0):
        """
        Draws a line of text on the info panel.
        :param text: The string to draw.
        :param color: BGR tuple for the text color.
        :param is_header: If True, formats the text as a header with a line below.
        :param indent: Horizontal indentation for the text.
        """
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
        """
        Draws the entire info panel onto the given frame.
        :param frame: The OpenCV frame (numpy array) to draw on.
        :param proc_fps: Current processing FPS.
        :param cam_res: Camera resolution (width, height).
        :param cam_fps: Camera frame rate.
        :param rec_status: Tuple (is_recording, recording_filepath).
        :param ref_status: Tuple (is_reference_enabled, is_reference_found, reference_marker_id).
        :param buoy_info: Dictionary of buoy summaries {buoy_id: {'num_markers': count}}.
        :param is_live: Boolean, True if in live mode, False for offline processing.
        """
        self.frame = frame
        panel = self.frame[:, :self.panel_width]
        # Darken the panel background
        cv2.addWeighted(panel, 0.5, np.zeros_like(panel), 0.5, 0, panel)
        self._reset_y_position()

        self._draw_text_line("PERFORMANCE", self.colors['white'], is_header=True)
        self._draw_text_line(f"Processing FPS: {proc_fps:.1f}", self.colors['cyan'])
        if is_live:
            self._draw_text_line(f"Camera: {cam_res[0]}x{cam_res[1]} @ {cam_fps:.1f} FPS", self.colors['cyan'])
        else:
            self._draw_text_line(f"Video Res: {cam_res[0]}x{cam_res[1]}", self.colors['cyan'])

        self._draw_text_line("SYSTEM STATUS", self.colors['white'], is_header=True)
        rec_on, rec_file = rec_status
        rec_file_basename = os.path.basename(rec_file) if rec_file else ""
        self._draw_text_line(f"Recording: {'ON' if rec_on else 'OFF'}", self.colors['green'] if rec_on else self.colors['red'])
        if rec_on and rec_file_basename:
            self._draw_text_line(f"  {rec_file_basename}", self.colors['orange'], indent=20)

        ref_on, ref_found, ref_id = ref_status
        self._draw_text_line(f"Reference: {'ENABLED' if ref_on else 'DISABLED'}", self.colors['green'] if ref_on else self.colors['red'])
        if ref_on:
            self._draw_text_line(f"  > ID {ref_id}: {'Found' if ref_found else 'Lost'}", self.colors['green'] if ref_found else self.colors['red'], indent=20)

        self._draw_text_line("BUOY DETECTION", self.colors['white'], is_header=True)
        if not buoy_info:
            self._draw_text_line("None", self.colors['yellow'])
        else:
            sorted_buoys = sorted(buoy_info.items())
            for buoy_id, info in sorted_buoys:
                num_markers = info.get('num_markers', 0)
                quality_color = self.colors['green'] if num_markers >= 3 else (self.colors['yellow'] if num_markers == 2 else self.colors['red'])
                
                # Draw a small circle indicator next to buoy ID
                circle_center = (self.x_margin + 12, self.y_pos - 12)
                cv2.circle(self.frame, circle_center, 10, quality_color, -1)
                cv2.circle(self.frame, circle_center, 10, self.colors['white'], self.thickness)
                self._draw_text_line(f"Buoy {buoy_id}: {num_markers} marker(s)", self.colors['white'], indent=40)

class PoseFilter:
    """
    A simple low-pass filter for pose data (translation and rotation).
    Uses exponential smoothing (alpha blending).
    """
    def __init__(self, alpha_translation=0.5, alpha_rotation=0.5):
        """
        Initializes the pose filter.
        :param alpha_translation: Smoothing factor for translation (0.0 to 1.0).
                                  0.0 means heavy smoothing, 1.0 means no smoothing.
        :param alpha_rotation: Smoothing factor for rotation (0.0 to 1.0).
        """
        self.alpha_t = np.clip(alpha_translation, 0, 1)
        self.alpha_r = np.clip(alpha_rotation, 0, 1)
        self.filtered_tvec = None
        self.filtered_quat = None # Store rotation as quaternion for spherical linear interpolation (SLERP)
        self.is_initialized = False

    def update(self, new_tvec, new_rvec):
        """
        Updates the filter with a new pose.
        :param new_tvec: New translation vector (3x1 numpy array).
        :param new_rvec: New rotation vector (3x1 numpy array).
        :return: Filtered translation and rotation vectors (3x1 numpy arrays).
        """
        new_tvec = np.asarray(new_tvec).flatten()
        try:
            new_rot = R.from_rotvec(np.asarray(new_rvec).flatten())
            new_quat = new_rot.as_quat()
        except ValueError:
            # If new_rvec is invalid, return previous filtered pose or raw input if not initialized
            if self.is_initialized:
                filtered_r = R.from_quat(self.filtered_quat)
                return self.filtered_tvec.reshape(3, 1), filtered_r.as_rotvec().reshape(3, 1)
            else:
                return np.asarray(new_tvec).reshape(3,1), np.asarray(new_rvec).reshape(3,1)

        if not self.is_initialized:
            self.filtered_tvec = new_tvec
            self.filtered_quat = new_quat
            self.is_initialized = True
        else:
            # Linear interpolation for translation
            self.filtered_tvec = self.alpha_t * new_tvec + (1.0 - self.alpha_t) * self.filtered_tvec

            # Spherical linear interpolation (SLERP) for rotation
            # Ensure quaternions are in the same hemisphere for shortest path interpolation
            if np.dot(self.filtered_quat, new_quat) < 0:
                new_quat = -new_quat
            interp_quat = (1.0 - self.alpha_r) * self.filtered_quat + self.alpha_r * new_quat
            # Re-normalize to ensure it's a unit quaternion
            self.filtered_quat = interp_quat / np.linalg.norm(interp_quat)

        filtered_r = R.from_quat(self.filtered_quat)
        filtered_rvec = filtered_r.as_rotvec().reshape(3, 1)
        return self.filtered_tvec.reshape(3, 1), filtered_rvec

    def reset(self):
        """Resets the filter, clearing any accumulated state."""
        self.is_initialized = False
        self.filtered_tvec = None
        self.filtered_quat = None

class RealTimePlotter:
    """
    Manages and updates real-time Matplotlib plots for buoy position and orientation.
    Converts plots to an OpenCV image for display.
    """
    def __init__(self, history_len, plot_width_px, video_height_px, buoy_ids_to_plot, is_relative_pose):
        """
        Initializes the plotter.
        :param history_len: Number of data points to keep in history for plotting.
        :param plot_width_px: Desired width of the plot image in pixels.
        :param video_height_px: Height of the video frame, used to match plot height.
        :param buoy_ids_to_plot: List of buoy IDs for which to plot data.
        :param is_relative_pose: Boolean, True if poses are relative to reference marker, False for camera frame.
        """
        self.history_len = history_len
        self.plot_width_px = plot_width_px
        self.plot_height_px = video_height_px
        self.dpi = 100 # Dots per inch for Matplotlib figure
        
        # Calculate figure size in inches
        fig_width_in = self.plot_width_px / self.dpi
        fig_height_in = self.plot_height_px / self.dpi

        self.fig, self.axes = plt.subplots(2, 1, figsize=(fig_width_in, fig_height_in), dpi=self.dpi, facecolor='#DDDDDD')
        self.ax_pos, self.ax_rot = self.axes # Separate axes for position and rotation

        self.start_time = None
        self.buoy_data = {} # Stores deque for each buoy's data
        self.buoy_lines = {} # Stores Line2D objects for updating plots

        # Use a colormap to assign distinct colors to each buoy
        cmap = matplotlib.colormaps['tab10']
        colors = cmap(np.linspace(0, 1, NUM_BUOYS)) # Assuming NUM_BUOYS is max number of buoys

        # Configure Position Plot
        pos_unit = "m (Rel)" if is_relative_pose else "m (Abs)"
        self.ax_pos.set_title("Buoy CoG Position", fontsize=PLOT_TITLE_FONTSIZE)
        self.ax_pos.set_ylabel(f"Position [{pos_unit}]", fontsize=PLOT_LABEL_FONTSIZE)
        self.ax_pos.grid(True, linestyle='--', alpha=0.6)
        self.ax_pos.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)

        # Configure Rotation Plot
        self.ax_rot.set_title("Buoy CoG Orientation (Stable Angles)", fontsize=PLOT_TITLE_FONTSIZE)
        self.ax_rot.set_ylabel("Rotation [deg]", fontsize=PLOT_LABEL_FONTSIZE)
        self.ax_rot.set_xlabel("Time [s]", fontsize=PLOT_LABEL_FONTSIZE)
        self.ax_rot.grid(True, linestyle='--', alpha=0.6)
        self.ax_rot.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)

        # Initialize data queues and plot lines for each buoy
        for i, buoy_id in enumerate(buoy_ids_to_plot):
            color = colors[i % len(colors)] # Cycle through colors if more buoys than colors
            label = f"B{buoy_id}"
            self.buoy_data[buoy_id] = {
                'time': deque(maxlen=history_len),
                'pos_x': deque(maxlen=history_len), 'pos_y': deque(maxlen=history_len), 'pos_z': deque(maxlen=history_len),
                'rot_x': deque(maxlen=history_len), 'rot_y': deque(maxlen=history_len), 'rot_z': deque(maxlen=history_len)
            }
            pos_lines = {
                'x': self.ax_pos.plot([], [], label=f'{label} X', color=color, linestyle='-', linewidth=PLOT_LINE_WIDTH)[0],
                'y': self.ax_pos.plot([], [], label=f'{label} Y', color=color, linestyle='--', linewidth=PLOT_LINE_WIDTH)[0],
                'z': self.ax_pos.plot([], [], label=f'{label} Z', color=color, linestyle=':', linewidth=PLOT_LINE_WIDTH)[0]
            }
            rot_lines = {
                'x': self.ax_rot.plot([], [], label=f'{label} Pitch', color=color, linestyle='-', linewidth=PLOT_LINE_WIDTH)[0],
                'y': self.ax_rot.plot([], [], label=f'{label} Inclination', color=color, linestyle='--', linewidth=PLOT_LINE_WIDTH)[0],
                'z': self.ax_rot.plot([], [], label=f'{label} Yaw', color=color, linestyle=':', linewidth=PLOT_LINE_WIDTH)[0]
            }
            self.buoy_lines[buoy_id] = {'pos': pos_lines, 'rot': rot_lines}

        # Add legends to plots
        self.ax_pos.legend(loc='upper left', fontsize=PLOT_LEGEND_FONTSIZE, ncol=max(1, len(buoy_ids_to_plot) // 3))
        self.ax_rot.legend(loc='upper left', fontsize=PLOT_LEGEND_FONTSIZE, ncol=max(1, len(buoy_ids_to_plot) // 3))
        
        # Adjust layout to prevent labels/titles from overlapping
        self.fig.tight_layout(pad=3.5)

    def _canvas_to_numpy(self):
        """Converts the Matplotlib figure to a NumPy array (OpenCV image)."""
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        
        img_rgba = np.asarray(buf)
        return cv2.cvtColor(img_rgba, cv2.COLOR_RGB2BGR)

    def update_and_get_plot_image(self, new_data_dict, timestamp):
        """
        Updates the plot data with new buoy poses and returns the plot as an image.
        :param new_data_dict: Dictionary of new data for each buoy.
                              Format: {buoy_id: {'pos': [x,y,z], 'rot': [pitch,inclination,yaw]}}
        :param timestamp: Current timestamp for the X-axis of the plot.
        :return: NumPy array representing the plot image (BGR format).
        """
        if self.start_time is None:
            self.start_time = timestamp
        current_rel_time = timestamp - self.start_time

        # Update data for each buoy
        for buoy_id, data in new_data_dict.items():
            if buoy_id not in self.buoy_data:
                continue # Skip if buoy ID not configured for plotting
            
            # Append new data to deques
            self.buoy_data[buoy_id]['time'].append(current_rel_time)
            self.buoy_data[buoy_id]['pos_x'].append(data['pos'][0])
            self.buoy_data[buoy_id]['pos_y'].append(data['pos'][1])
            self.buoy_data[buoy_id]['pos_z'].append(data['pos'][2])
            
            # Rotation data (stable angles: pitch, inclination, yaw)
            stable_rot = data['rot']
            self.buoy_data[buoy_id]['rot_x'].append(stable_rot[0]) # Pitch
            self.buoy_data[buoy_id]['rot_y'].append(stable_rot[1]) # Inclination
            self.buoy_data[buoy_id]['rot_z'].append(stable_rot[2]) # Yaw

        # Update plot lines with new data
        for buoy_id, lines_dict in self.buoy_lines.items():
            time_data = self.buoy_data[buoy_id]['time']
            if not time_data:
                continue # Skip if no data for this buoy yet

            lines_dict['pos']['x'].set_data(time_data, self.buoy_data[buoy_id]['pos_x'])
            lines_dict['pos']['y'].set_data(time_data, self.buoy_data[buoy_id]['pos_y'])
            lines_dict['pos']['z'].set_data(time_data, self.buoy_data[buoy_id]['pos_z'])

            lines_dict['rot']['x'].set_data(time_data, self.buoy_data[buoy_id]['rot_x'])
            lines_dict['rot']['y'].set_data(time_data, self.buoy_data[buoy_id]['rot_y'])
            lines_dict['rot']['z'].set_data(time_data, self.buoy_data[buoy_id]['rot_z'])

        # Auto-scale axes based on current data limits
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        
        # Render the plot and return as an image
        return self._canvas_to_numpy()

## END SECTION: HELPER CLASSES

## SECTION: CONFIGURATION & CALIBRATION MANAGEMENT

def save_settings():
    """Saves the current application settings to a JSON file."""
    settings_to_save = {
        "CAMERA_INDEX": CAMERA_INDEX,
        "CAMERA_RESOLUTION": CAMERA_RESOLUTION,
        "CAMERA_FPS": CAMERA_FPS,
        "REVERSE_CAMERA_DISPLAY": REVERSE_CAMERA_DISPLAY,
        "SHOW_REALTIME_PLOTS": SHOW_REALTIME_PLOTS,
        "RECORD_VIDEO": RECORD_VIDEO,
        "MARKER_SIZE_METERS": MARKER_SIZE_METERS,
        "REFERENCE_MARKER_ID": REFERENCE_MARKER_ID,
        "USE_REFERENCE_MARKER": USE_REFERENCE_MARKER,
        "ENABLE_REFERENCE_FILTER": ENABLE_REFERENCE_FILTER,
        "REF_FILTER_ALPHA_T": REF_FILTER_ALPHA_T,
        "REF_FILTER_ALPHA_R": REF_FILTER_ALPHA_R,
        "ENABLE_LED_DETECTION": ENABLE_LED_DETECTION,
        "LED_DETECTION_ROI": LED_DETECTION_ROI,
        "LED_OFF_BASELINE_BRIGHTNESS": LED_OFF_BASELINE_BRIGHTNESS,
        "LED_BRIGHTNESS_THRESHOLD": LED_BRIGHTNESS_THRESHOLD,
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
    """Loads application settings from a JSON file."""
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, SETTINGS_FILE)
    if not os.path.exists(filepath):
        print(f"INFO: Settings file not found at {filepath}. Using default values.")
        return
    try:
        with open(filepath, 'r') as f:
            settings = json.load(f)
        
        # Update global variables with loaded settings, providing defaults if a key is missing
        global CAMERA_INDEX, CAMERA_RESOLUTION, CAMERA_FPS, REVERSE_CAMERA_DISPLAY
        global SHOW_REALTIME_PLOTS, RECORD_VIDEO, MARKER_SIZE_METERS
        global REFERENCE_MARKER_ID, USE_REFERENCE_MARKER, ENABLE_REFERENCE_FILTER, REF_FILTER_ALPHA_T, REF_FILTER_ALPHA_R
        global ENABLE_LED_DETECTION, LED_DETECTION_ROI, LED_OFF_BASELINE_BRIGHTNESS
        global LED_BRIGHTNESS_THRESHOLD, LOG_ALL_MARKER_DATA

        CAMERA_INDEX = settings.get("CAMERA_INDEX", CAMERA_INDEX)
        # Ensure CAMERA_RESOLUTION is a tuple
        CAMERA_RESOLUTION = tuple(settings.get("CAMERA_RESOLUTION", CAMERA_RESOLUTION))
        CAMERA_FPS = settings.get("CAMERA_FPS", CAMERA_FPS)
        REVERSE_CAMERA_DISPLAY = settings.get("REVERSE_CAMERA_DISPLAY", REVERSE_CAMERA_DISPLAY)
        SHOW_REALTIME_PLOTS = settings.get("SHOW_REALTIME_PLOTS", SHOW_REALTIME_PLOTS)
        RECORD_VIDEO = settings.get("RECORD_VIDEO", RECORD_VIDEO)
        MARKER_SIZE_METERS = settings.get("MARKER_SIZE_METERS", MARKER_SIZE_METERS)
        REFERENCE_MARKER_ID = settings.get("REFERENCE_MARKER_ID", REFERENCE_MARKER_ID)
        USE_REFERENCE_MARKER = settings.get("USE_REFERENCE_MARKER", USE_REFERENCE_MARKER)
        ENABLE_REFERENCE_FILTER = settings.get("ENABLE_REFERENCE_FILTER", ENABLE_REFERENCE_FILTER)
        REF_FILTER_ALPHA_T = settings.get("REF_FILTER_ALPHA_T", REF_FILTER_ALPHA_T)
        REF_FILTER_ALPHA_R = settings.get("REF_FILTER_ALPHA_R", REF_FILTER_ALPHA_R)
        ENABLE_LED_DETECTION = settings.get("ENABLE_LED_DETECTION", ENABLE_LED_DETECTION)
        LED_DETECTION_ROI = settings.get("LED_DETECTION_ROI", LED_DETECTION_ROI)
        # Ensure LED_DETECTION_ROI is a tuple if loaded as list
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
    """
    Loads camera calibration parameters (camera matrix and distortion coefficients)
    from a NumPy .npz file.
    :return: Tuple (camera_matrix, dist_coeffs). Returns default if file not found or error.
    """
    calibration_path = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, CALIBRATION_FILE)
    try:
        data = np.load(calibration_path)
        print(f"Camera calibration loaded from: {calibration_path}")
        return data['camera_matrix'], data['dist_coeffs']
    except Exception:
        print(f"WARN: Calibration file not found or error loading. Using defaults: {calibration_path}")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

def get_video_writer(frame_size, full_video_path, fps=None):
    """
    Initializes and returns an OpenCV VideoWriter object.
    Attempts to use the primary codec, then falls back to 'XVID' if necessary.
    :param frame_size: Tuple (width, height) of the video frames.
    :param full_video_path: Full path including filename and extension for the output video.
    :param fps: Frames per second for the output video. If None, uses VIDEO_FPS.
    :return: cv2.VideoWriter object or None if creation fails.
    """
    output_fps = fps if fps is not None and fps > 0 else VIDEO_FPS
    codec_to_try = VIDEO_CODEC
    writer = None
    
    # Try preferred codec
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_to_try)
        writer = cv2.VideoWriter(full_video_path, fourcc, output_fps, frame_size)
        if writer.isOpened():
            print(f"Video writer created for: {full_video_path} with codec '{codec_to_try}'.")
            return writer
    except Exception as e:
        print(f"ERROR: Initializing writer with codec '{codec_to_try}': {e}")

    # Fallback to 'XVID'
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
    """
    Loads buoy marker relative geometry calibrations from a JSON file.
    These are the relative poses of markers on a buoy with respect to its "X0" marker.
    :return: Dictionary of calibrations. Returns empty dict if file not found or error.
    """
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, BUOY_GEOMETRY_CALIBRATION_FILE)
    if not os.path.exists(filepath):
        print(f"INFO: Buoy geometry calibration file not found: {filepath}. Using defaults.")
        return {}
    try:
        with open(filepath, 'r') as f:
            calibrations = json.load(f)
        
        # Convert list representations back to NumPy arrays
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
    """
    Saves buoy marker relative geometry calibrations to a JSON file.
    Converts NumPy arrays to lists for JSON serialization.
    :param calibrations: Dictionary of buoy calibrations.
    """
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, BUOY_GEOMETRY_CALIBRATION_FILE)
    serializable_calibrations = {}
    try:
        for buoy_id, markers_data in calibrations.items():
            serializable_calibrations[str(buoy_id)] = {} # Ensure buoy_id is string key
            for marker_id, pose_data in markers_data.items():
                s_marker_data = {}
                # Convert NumPy arrays to lists for JSON serialization
                if 't_rel_to_X0' in pose_data and isinstance(pose_data['t_rel_to_X0'], np.ndarray):
                    s_marker_data['t_rel_to_X0'] = pose_data['t_rel_to_X0'].tolist()
                else:
                    s_marker_data['t_rel_to_X0'] = None # Store None if not available or invalid
                
                if 'rvec_rel_to_X0' in pose_data and isinstance(pose_data['rvec_rel_to_X0'], np.ndarray):
                    s_marker_data['rvec_rel_to_X0'] = pose_data['rvec_rel_to_X0'].tolist()
                else:
                    s_marker_data['rvec_rel_to_X0'] = None
                
                serializable_calibrations[str(buoy_id)][str(marker_id)] = s_marker_data # Ensure marker_id is string key
        
        with open(filepath, 'w') as f:
            json.dump(serializable_calibrations, f, indent=4)
        print(f"Saved buoy geometry calibrations to: {filepath}")
    except Exception as e:
        print(f"ERROR: Saving buoy geometry calibrations to {filepath}: {e}")

# Global variables to store the initialized buoy configurations
ALL_BUOY_GEOMETRY_CALIBRATIONS = {}
MARKER_BUOY_GEOMETRY_CONFIG = {} # Stores marker poses relative to X0 in the BRF (Buoy Reference Frame)
BUOY_COG_OFFSETS_IN_BRF_CONFIG = {} # Stores CoG offsets for each buoy in its BRF

def initialize_buoy_configs():
    """
    Initializes or re-initializes buoy configurations based on loaded calibrations
    and default marker relative geometry.
    This function should be called after loading settings or calibration files.
    """
    global MARKER_BUOY_GEOMETRY_CONFIG, BUOY_COG_OFFSETS_IN_BRF_CONFIG, ALL_BUOY_GEOMETRY_CALIBRATIONS

    # 1. Load all stored buoy geometry calibrations
    ALL_BUOY_GEOMETRY_CALIBRATIONS = load_buoy_geometry_calibrations()

    temp_marker_buoy_geometry_config = {}
    for buoy_id, marker_ids_in_buoy in BUOY_ID_TO_MARKER_IDS_MAP.items():
        buoy_geom = {}
        buoy_id_str = str(buoy_id)
        id_X0 = marker_ids_in_buoy[0] # The first marker is designated as X0, the buoy's origin

        # X0's pose in its own frame (BRF) is identity (translation 0, rotation identity)
        buoy_geom[id_X0] = {'t_BRF_Xi': np.zeros((3, 1), dtype=np.float32), 
                            'R_BRF_Xi': np.eye(3, dtype=np.float32)}

        # Iterate through other markers on the buoy (Xi)
        for i_role, marker_id_actual in enumerate(marker_ids_in_buoy):
            if i_role == 0: # Skip X0 as it's already set
                continue

            marker_id_str = str(marker_id_actual)
            use_calibrated = False
            
            # Check if a specific calibration exists for this buoy and marker pair
            if buoy_id_str in ALL_BUOY_GEOMETRY_CALIBRATIONS and \
               marker_id_str in ALL_BUOY_GEOMETRY_CALIBRATIONS[buoy_id_str]:
                calib_data = ALL_BUOY_GEOMETRY_CALIBRATIONS[buoy_id_str][marker_id_str]
                # Ensure calibration data is valid (not None for tvec/rvec)
                if calib_data and calib_data.get('t_rel_to_X0') is not None and \
                   calib_data.get('rvec_rel_to_X0') is not None:
                    use_calibrated = True

            if use_calibrated:
                # Use calibrated relative pose
                t_calib = np.array(calib_data['t_rel_to_X0'], dtype=np.float32).reshape(3, 1)
                R_calib, _ = cv2.Rodrigues(np.array(calib_data['rvec_rel_to_X0'], dtype=np.float32))
                buoy_geom[marker_id_actual] = {'t_BRF_Xi': t_calib, 'R_BRF_Xi': R_calib}
            elif i_role in MARKER_RELATIVE_GEOMETRY:
                # Fallback to ideal/predefined geometry if no specific calibration
                ideal_geom = MARKER_RELATIVE_GEOMETRY[i_role]
                R_ideal, _ = cv2.Rodrigues(ideal_geom['rel_rvec_to_X0'])
                buoy_geom[marker_id_actual] = {'t_BRF_Xi': ideal_geom['rel_tvec_to_X0'].reshape(3, 1), 
                                                'R_BRF_Xi': R_ideal}
            else:
                # If no calibration and no ideal geometry, mark as NaN (invalid)
                buoy_geom[marker_id_actual] = {'t_BRF_Xi': np.full((3, 1), np.nan), 
                                                'R_BRF_Xi': np.full((3, 3), np.nan)}
        temp_marker_buoy_geometry_config[buoy_id] = buoy_geom
    
    # Update global configurations
    MARKER_BUOY_GEOMETRY_CONFIG.clear()
    MARKER_BUOY_GEOMETRY_CONFIG.update(temp_marker_buoy_geometry_config)

    # CoG offset is typically fixed for a buoy design, so we apply the global constant
    BUOY_COG_OFFSETS_IN_BRF_CONFIG.clear()
    BUOY_COG_OFFSETS_IN_BRF_CONFIG.update({buoy_id: BUOY_COG_OFFSET_VECTOR_ASSEMBLY_FRAME.reshape(3, 1) 
                                           for buoy_id in BUOY_ID_TO_MARKER_IDS_MAP.keys()})

    print("Buoy configurations initialized/re-initialized.")

# Initialize buoy configurations on script startup
initialize_buoy_configs()

## END SECTION: CONFIGURATION & CALIBRATION MANAGEMENT

## SECTION: POSE UTILITIES

def rotation_vector_to_euler(rvec):
    """
    Converts a rotation vector (Rodrigues) to Euler angles (XYZ, degrees) and quaternion.
    :param rvec: 3x1 or 1x3 rotation vector.
    :return: Tuple (euler_angles_deg (np.array 1x3), quaternion (np.array 1x4)).
             Returns zeros/identity if rvec is None or invalid.
    """
    if rvec is None:
        return np.zeros(3), R.identity().as_quat()
    
    rvec_np = np.asarray(rvec).reshape(3)
    if np.allclose(rvec_np, 0): # Handle zero rotation vector case
        return np.zeros(3), R.identity().as_quat()
        
    try:
        rotation = R.from_rotvec(rvec_np)
        euler_angles_deg = rotation.as_euler('xyz', degrees=True)
        quaternion = rotation.as_quat()
        return euler_angles_deg, quaternion
    except Exception:
        # Fallback for any other conversion errors
        print(f"Warning: Could not convert rvec {rvec_np} to Euler/Quaternion. Returning zeros/identity.")
        return np.zeros(3), R.identity().as_quat()

def get_stable_orientation_angles_from_matrix(rot_matrix):
    """
    Calculates "stable" orientation angles (yaw, pitch, inclination) from a rotation matrix.
    - Yaw: Rotation around world Z-axis (heading).
    - Pitch: Rotation around body X-axis.
    - Inclination: Angle between body Z-axis and world Z-axis.
    These are generally more intuitive than standard Euler angles for buoy orientation.
    :param rot_matrix: 3x3 rotation matrix from body frame to world frame.
    :return: Tuple (yaw_deg, pitch_deg, inclination_deg).
    """
    # Define body axes in the body frame
    body_x_axis = np.array([1., 0., 0.])
    body_z_axis = np.array([0., 0., 1.])

    # Transform body axes to the world frame
    body_x_in_world = rot_matrix @ body_x_axis
    body_z_in_world = rot_matrix @ body_z_axis

    # Yaw (heading): Angle of the projected body X-axis on the world XY-plane
    yaw = np.rad2deg(np.arctan2(body_x_in_world[1], body_x_in_world[0]))

    # Pitch: Rotation around the body X-axis. Derived from the Z-component of the X-axis in world frame.
    # Clip to avoid arcsin errors due to floating point inaccuracies
    pitch = np.rad2deg(np.arcsin(-np.clip(body_x_in_world[2], -1.0, 1.0)))

    # Inclination (roll-like but independent of yaw/pitch): Angle between world Z and body Z.
    # Derived from the dot product of world Z (0,0,1) and body Z in world (which is body_z_in_world[2]).
    cos_inclination = np.clip(body_z_in_world[2], -1.0, 1.0)
    inclination = np.rad2deg(np.arccos(cos_inclination))

    return yaw, pitch, inclination

def transform_pose_to_reference(rvec_obj_cam, tvec_obj_cam, rvec_ref_cam, tvec_ref_cam):
    """
    Transforms an object's pose (rvec, tvec relative to camera) to be relative
    to a reference marker's pose (rvec_ref_cam, tvec_ref_cam).
    
    This calculates T_ref_obj = T_ref_cam @ T_cam_obj
    where T_cam_obj is the pose of object in camera frame,
    and T_ref_cam is the pose of camera in reference frame.
    
    :param rvec_obj_cam: Rotation vector of object relative to camera.
    :param tvec_obj_cam: Translation vector of object relative to camera.
    :param rvec_ref_cam: Rotation vector of reference marker relative to camera.
    :param tvec_ref_cam: Translation vector of reference marker relative to camera.
    :return: Tuple (rvec_ref_obj, tvec_ref_obj), the object's pose relative to reference marker.
    """
    # Convert rvecs to rotation matrices
    R_obj_cam, _ = cv2.Rodrigues(np.asarray(rvec_obj_cam))
    R_ref_cam, _ = cv2.Rodrigues(np.asarray(rvec_ref_cam))

    # Create transformation matrices from camera to object and camera to reference
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R_obj_cam
    T_cam_obj[:3, 3] = np.asarray(tvec_obj_cam).flatten()

    T_cam_ref = np.eye(4)
    T_cam_ref[:3, :3] = R_ref_cam
    T_cam_ref[:3, 3] = np.asarray(tvec_ref_cam).flatten()

    # Invert T_cam_ref to get T_ref_cam (pose of camera in reference frame)
    T_ref_cam = np.linalg.inv(T_cam_ref)

    # Calculate T_ref_obj (pose of object in reference frame)
    T_ref_obj = T_ref_cam @ T_cam_obj

    # Extract rotation matrix and translation vector
    R_ref_obj = T_ref_obj[:3, :3]
    t_ref_obj = T_ref_obj[:3, 3]

    # Convert rotation matrix back to rotation vector
    rvec_ref_obj, _ = cv2.Rodrigues(R_ref_obj)

    return rvec_ref_obj.reshape(3, 1), t_ref_obj.reshape(3, 1)

def average_quaternions_weighted(quaternions, weights):
    """
    Computes a weighted average of quaternions using Eigen-decomposition.
    This method handles averaging orientations correctly.
    :param quaternions: List of quaternions (each as a 4-element NumPy array).
    :param weights: List of corresponding weights.
    :return: Averaged quaternion (4-element NumPy array).
    """
    if not quaternions:
        return R.identity().as_quat() # Return identity quaternion if no inputs
    if len(quaternions) == 1:
        return quaternions[0]

    # Align quaternions to be in the same hemisphere to prevent cancellation
    q_ref = quaternions[0]
    for i in range(1, len(quaternions)):
        if np.dot(q_ref, quaternions[i]) < 0:
            quaternions[i] *= -1

    # Form the M matrix for Eigen-decomposition
    # M = sum(wi * qi * qi^T) for all i
    M = np.zeros((4, 4))
    for q_i, w_i in zip(quaternions, weights):
        q_i = q_i.reshape(4, 1) # Make it a column vector
        M += w_i * (q_i @ q_i.T) # Outer product weighted by w_i

    # The dominant eigenvector of M corresponds to the average quaternion
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg_quat = eigenvectors[:, np.argmax(eigenvalues)]

    return avg_quat / np.linalg.norm(avg_quat) # Normalize to ensure it's a unit quaternion

## END SECTION: POSE UTILITIES

## SECTION: CORE TRACKING LOGIC

def _detect_and_estimate_markers(gray_frame, aruco_dict, aruco_parameters, marker_size_m, camera_matrix, dist_coeffs):
    """
    Detects ArUco markers in a grayscale frame and estimates their poses.
    :return: A tuple (corners, ids, rvecs, tvecs, refined_corners_list).
             rvecs/tvecs are lists, refined_corners_list are per-marker arrays.
             Returns (None, None, None, None, None) if no markers detected.
    """
    corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)
    
    if ids is None or len(ids) == 0:
        return None, None, None, None, None

    # Refine corner locations for sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners_list = [cv2.cornerSubPix(gray_frame, c.astype(np.float32), (5,5), (-1,-1), criteria) for c in corners]

    # Estimate pose for each detected marker
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(refined_corners_list, marker_size_m, camera_matrix, dist_coeffs)
    
    return corners, ids, rvecs, tvecs, refined_corners_list

def _process_reference_marker(ids, rvecs_all, tvecs_all, refined_corners_list, camera_matrix, dist_coeffs, marker_size_m, ref_pose_filter, annotated_frame):
    """
    Processes the reference marker, applying a filter if enabled, and draws its axes.
    :return: Tuple (ref_marker_detected, ref_rvec, ref_tvec)
    """
    ref_marker_detected = False
    ref_rvec, ref_tvec = None, None

    try:
        ref_idx = np.where(ids.flatten() == REFERENCE_MARKER_ID)[0][0]
        ref_marker_detected = True
        raw_ref_rvec = rvecs_all[ref_idx].reshape(3,1)
        raw_ref_tvec = tvecs_all[ref_idx].reshape(3,1)

        # Apply filter if enabled
        if ENABLE_REFERENCE_FILTER and ref_pose_filter:
            ref_tvec, ref_rvec = ref_pose_filter.update(raw_ref_tvec, raw_ref_rvec)
        else:
            ref_rvec, ref_tvec = raw_ref_rvec, raw_ref_tvec
        
        # Draw axes for the reference marker
        cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs, ref_rvec, ref_tvec, marker_size_m, AXES_THICKNESS)
    except IndexError:
        # Reference marker not found in this frame
        if ref_pose_filter:
            ref_pose_filter.reset() # Reset filter if marker is lost
        pass # No reference marker detected
    except Exception as e:
        print(f"Error processing reference marker (ID {REFERENCE_MARKER_ID}): {e}")
        if ref_pose_filter:
            ref_pose_filter.reset()
        ref_marker_detected = False

    return ref_marker_detected, ref_rvec, ref_tvec

def _get_all_marker_data(ids, rvecs_all, tvecs_all, refined_corners_list, use_reference_marker, ref_marker_detected, ref_rvec, ref_tvec):
    """
    Compiles data for all detected markers, including relative poses if reference is used.
    :return: Dictionary {marker_id: {'cam_tvec', 'cam_rvec', 'cam_quat', 'corners', 'rel_tvec', 'rel_rvec', 'rel_quat'}}
    """
    all_markers_data_this_frame = {}
    for i, marker_id_arr in enumerate(ids):
        marker_id = marker_id_arr[0]
        r_cam_marker, t_cam_marker = rvecs_all[i].reshape(3,1), tvecs_all[i].reshape(3,1)
        
        # Get Euler angles and quaternion for camera frame pose
        _, quat_cam_marker = rotation_vector_to_euler(r_cam_marker)
        
        marker_data = {
            'cam_tvec': t_cam_marker,
            'cam_rvec': r_cam_marker,
            'cam_quat': quat_cam_marker,
            'corners': refined_corners_list[i]
        }

        # Calculate relative pose if reference marker is used and detected
        if use_reference_marker and ref_marker_detected:
            try:
                r_rel, t_rel = transform_pose_to_reference(r_cam_marker, t_cam_marker, ref_rvec, ref_tvec)
                _, quat_rel = rotation_vector_to_euler(r_rel)
                marker_data['rel_tvec'] = t_rel
                marker_data['rel_rvec'] = r_rel
                marker_data['rel_quat'] = quat_rel
            except Exception as e:
                print(f"Warning: Could not calculate relative pose for marker {marker_id}: {e}")
                # Don't add relative pose data if calculation fails
        
        all_markers_data_this_frame[marker_id] = marker_data
    return all_markers_data_this_frame

def _log_individual_marker_data(all_markers_data_this_frame, log_date_str, log_time_str, individual_marker_log_entries, use_reference_marker):
    """
    Generates log entries for individual marker poses.
    """
    for marker_id, data in all_markers_data_this_frame.items():
        log_row = [log_date_str, log_time_str, marker_id]
        
        # Camera frame pose
        cam_euler, cam_quat = rotation_vector_to_euler(data['cam_rvec'])
        log_row.extend([f"{v:.6f}" for v in data['cam_tvec'].flatten()])
        log_row.extend([f"{v:.3f}" for v in cam_euler.flatten()])
        log_row.extend([f"{v:.6f}" for v in cam_quat])

        # Relative pose (if applicable)
        if use_reference_marker:
            if 'rel_tvec' in data:
                rel_euler, rel_quat = rotation_vector_to_euler(data['rel_rvec'])
                log_row.extend([f"{v:.6f}" for v in data['rel_tvec'].flatten()])
                log_row.extend([f"{v:.3f}" for v in rel_euler.flatten()])
                log_row.extend([f"{v:.6f}" for v in rel_quat])
            else:
                log_row.extend(["NaN"] * 10) # 3 tvec, 3 euler, 4 quat = 10 NaNs
        individual_marker_log_entries.append(log_row)

def _estimate_buoy_cog_poses(all_markers_data_this_frame, camera_matrix, dist_coeffs, marker_size_m, use_reference_marker, ref_marker_detected, ref_rvec, ref_tvec, buoy_id_to_marker_ids_map, marker_buoy_geometry_config, buoy_cog_offsets_in_brf_config, log_date_str, log_time_str, led_state_this_frame, cog_log_entries, plot_data_for_frame, annotated_frame, positions_history):
    """
    Estimates the Center of Gravity (CoG) pose for each buoy.
    Performs weighted averaging of marker poses on each buoy.
    Draws axes and trajectory, and generates log data and plot data.
    """
    buoy_pose_estimates = {} # Stores temporary estimates for each buoy from its markers
    marker_obj_pts = np.array([[-marker_size_m/2, marker_size_m/2, 0], 
                               [marker_size_m/2, marker_size_m/2, 0], 
                               [marker_size_m/2, -marker_size_m/2, 0], 
                               [-marker_size_m/2, -marker_size_m/2, 0]], dtype=np.float32)

    # Accumulate estimates for each buoy from its constituent markers
    for marker_id, data in all_markers_data_this_frame.items():
        buoy_id = MARKER_ID_TO_BUOY_ID_MAP.get(marker_id)
        if buoy_id is None: # Not a buoy marker
            continue
        
        buoy_geom_map = marker_buoy_geometry_config.get(buoy_id)
        if not buoy_geom_map or marker_id not in buoy_geom_map:
            continue # Buoy geometry config missing for this buoy/marker
        
        geom_Xi_in_BRF = buoy_geom_map[marker_id]
        if np.isnan(geom_Xi_in_BRF['t_BRF_Xi']).any():
            continue # Invalid/uncalibrated marker geometry
        
        t_BRF_Xi, R_BRF_Xi = geom_Xi_in_BRF['t_BRF_Xi'], geom_Xi_in_BRF['R_BRF_Xi']
        R_cam_Xi, _ = cv2.Rodrigues(data['cam_rvec'])

        # Calculate the transformation from Camera to Buoy Reference Frame (BRF) via marker Xi
        # T_cam_BRF = T_cam_Xi @ T_Xi_BRF
        # T_Xi_BRF is the inverse of T_BRF_Xi
        R_Xi_BRF = R_BRF_Xi.T
        t_Xi_BRF = -R_Xi_BRF @ t_BRF_Xi

        R_cam_BRF_est = R_cam_Xi @ R_Xi_BRF
        t_cam_BRF_est = (R_cam_Xi @ t_Xi_BRF) + data['cam_tvec']

        # Calculate reprojection error for confidence weighting
        projected_corners, _ = cv2.projectPoints(marker_obj_pts, data['cam_rvec'], data['cam_tvec'], camera_matrix, dist_coeffs)
        error = cv2.norm(data['corners'].reshape(-1,2), projected_corners.reshape(-1,2), cv2.NORM_L2)
        confidence = 1.0 / (1.0 + error**2 + 1e-9) # Higher confidence for lower error

        buoy_pose_estimates.setdefault(buoy_id, []).append({'R_est': R_cam_BRF_est, 't_est': t_cam_BRF_est, 'confidence': confidence})

    detected_buoys_summary = {}
    for buoy_id, estimates in buoy_pose_estimates.items():
        if not estimates:
            continue

        detected_buoys_summary[buoy_id] = {'num_markers': len(estimates)}

        # Fuse multiple marker estimates for the buoy's BRF pose
        tvecs_to_avg = [est['t_est'] for est in estimates]
        quats_to_avg = [R.from_matrix(est['R_est']).as_quat() for est in estimates]
        weights = [est['confidence'] for est in estimates]

        # Weighted average for translation
        tvecs_np = np.hstack(tvecs_to_avg) # Stack into a (3, N) array
        weights_np = np.array(weights).reshape(1, -1) # Make weights (1, N) for broadcasting
        fused_t_cam_BRF = (np.sum(tvecs_np * weights_np, axis=1) / np.sum(weights_np)).reshape(3,1)

        # Weighted average for rotation using quaternion averaging
        fused_quat_cam_BRF = average_quaternions_weighted(quats_to_avg, weights)
        fused_R_cam_BRF = R.from_quat(fused_quat_cam_BRF).as_matrix()

        # Apply CoG offset to get final buoy CoG pose in camera frame
        cog_offset_in_brf = buoy_cog_offsets_in_brf_config.get(buoy_id, np.zeros((3,1)))
        t_cog_cam = fused_t_cam_BRF + (fused_R_cam_BRF @ cog_offset_in_brf)
        R_cog_cam = fused_R_cam_BRF
        r_cog_cam, _ = cv2.Rodrigues(R_cog_cam)

        # Prepare poses for logging and plotting
        final_R_for_log = R_cog_cam
        final_t_for_log = t_cog_cam
        is_relative_log = False

        if use_reference_marker and ref_marker_detected:
            try:
                # Convert buoy CoG pose to be relative to the reference marker
                # T_ref_Buoy_COG = T_ref_Cam @ T_Cam_Buoy_COG
                R_ref_cam, _ = cv2.Rodrigues(ref_rvec)
                final_R_for_log = R_ref_cam.T @ R_cog_cam
                final_t_for_log = R_ref_cam.T @ (t_cog_cam - ref_tvec)
                is_relative_log = True
            except Exception as e:
                print(f"Error transforming CoG to relative frame for buoy {buoy_id}: {e}")
                # Keep absolute pose if relative transformation fails
        
        # Calculate Euler angles and stable angles for logging and plotting
        r_for_log, _ = cv2.Rodrigues(final_R_for_log)
        euler_for_log, quat_for_log = rotation_vector_to_euler(r_for_log)
        stable_angles = get_stable_orientation_angles_from_matrix(final_R_for_log)
        
        # Plot data: Pitch, Inclination, Yaw are typically more stable than XYZ Euler
        plot_data_for_frame[buoy_id] = {'pos': final_t_for_log.flatten(), 
                                         'rot': np.array([stable_angles[1], stable_angles[2], stable_angles[0]])} # Pitch, Inclination, Yaw

        # Create log entry for buoy CoG
        log_entry = [log_date_str, log_time_str, buoy_id, 1 if led_state_this_frame else 0]
        if use_reference_marker:
            log_entry.append("1" if is_relative_log else "0") # Indicate if relative pose was computed successfully
            if is_relative_log:
                log_entry.extend([f"{v:.6f}" for v in final_t_for_log.flatten()])
                log_entry.extend([f"{v:.3f}" for v in euler_for_log.flatten()])
                log_entry.extend([f"{v:.6f}" for v in quat_for_log])
            else:
                log_entry.extend(["NaN"] * 10) # 3 tvec, 3 euler, 4 quat = 10 NaNs if relative failed
        
        # Always log absolute pose for comparison/fallback
        abs_euler, abs_quat = rotation_vector_to_euler(r_cog_cam)
        log_entry.extend([f"{v:.6f}" for v in t_cog_cam.flatten()])
        log_entry.extend([f"{v:.3f}" for v in abs_euler.flatten()])
        log_entry.extend([f"{v:.6f}" for v in abs_quat])
        cog_log_entries.append(log_entry)

        # Visualize buoy CoG
        try:
            # Project CoG origin to image plane
            imgpts_cog, _ = cv2.projectPoints(np.zeros((1,3)), r_cog_cam, t_cog_cam, camera_matrix, dist_coeffs)
            center_px = tuple(imgpts_cog[0,0].astype(int))
            
            # Update trajectory history and draw it
            positions_history.setdefault(buoy_id, deque(maxlen=TRAJECTORY_LENGTH)).append(center_px)
            
            # Draw CoG axes
            cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs, r_cog_cam, t_cog_cam, marker_size_m * 0.8, AXES_THICKNESS)
            
            # Draw CoG position/orientation text
            prefix = "Rel" if is_relative_log else "Abs"
            y_off, y_step = -80, int(35 * OVERLAY_FONT_SCALE) # Offset and line spacing for text
            
            cv2.putText(annotated_frame, f"Buoy {buoy_id} CoG", (center_px[0] + 20, center_px[1] + y_off), 
                        cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, BUOY_COG_INFO_COLOR, OVERLAY_THICKNESS)
            y_off += y_step
            cv2.putText(annotated_frame, f"{prefix} P:({final_t_for_log[0,0]:.2f},{final_t_for_log[1,0]:.2f},{final_t_for_log[2,0]:.2f})", 
                        (center_px[0] + 20, center_px[1] + y_off), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.8, BUOY_COG_INFO_COLOR, OVERLAY_THICKNESS)
            y_off += y_step
            cv2.putText(annotated_frame, f"{prefix} R:({euler_for_log[0]:.1f},{euler_for_log[1]:.1f},{euler_for_log[2]:.1f})", 
                        (center_px[0] + 20, center_px[1] + y_off), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.8, BUOY_COG_INFO_COLOR, OVERLAY_THICKNESS)
        except Exception as e:
            print(f"Error during CoG visualization for buoy {buoy_id}: {e}")
    
    return detected_buoys_summary

def _draw_trajectories(annotated_frame, positions_history):
    """Draws historical trajectories for all buoys."""
    for buoy_id_hist, pos_list_hist in positions_history.items():
        if len(pos_list_hist) > 1:
            pts = np.array(pos_list_hist, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts], isClosed=False, color=TRAJECTORY_COLOR, thickness=TRAJECTORY_THICKNESS)

def _perform_led_detection(color_frame, led_detection_roi, led_off_baseline, led_brightness_threshold):
    """
    Performs LED detection based on a defined ROI and brightness threshold.
    :param color_frame: The current color frame.
    :param led_detection_roi: Tuple (x, y, w, h) of the ROI.
    :param led_off_baseline: The baseline brightness when the LED is off.
    :param led_brightness_threshold: The threshold above baseline to consider LED on.
    :return: Tuple (led_state_this_frame, roi_color, status_text)
    """
    led_state_this_frame = False
    roi_color = (128, 128, 128) # Default gray for OFF or error
    status_text = "LED: N/A"

    if led_detection_roi is None:
        return led_state_this_frame, roi_color, "LED: No ROI"

    try:
        x, y, w, h = [int(v) for v in led_detection_roi]
        # Validate ROI dimensions
        if y + h > color_frame.shape[0] or x + w > color_frame.shape[1] or w <= 0 or h <= 0:
            status_text = "LED: ROI Invalid"
            return led_state_this_frame, roi_color, status_text
        
        led_roi = color_frame[y:y+h, x:x+w]
        # Check red channel brightness (assuming red LED)
        current_red_mean = np.mean(led_roi[:, :, 2]) # Index 2 for Red channel in BGR
        
        if current_red_mean > (led_off_baseline + led_brightness_threshold):
            led_state_this_frame = True
            roi_color = (0, 255, 0) # Green if ON
            status_text = "LED: ON"
        else:
            roi_color = (128, 128, 128) # Gray if OFF
            status_text = "LED: OFF"
        
        # Draw ROI rectangle on frame
        cv2.rectangle(color_frame, (x, y), (x + w, y + h), roi_color, 2)
        cv2.putText(color_frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.7, roi_color, OVERLAY_THICKNESS)

    except Exception as e:
        print(f"Warning: Error during LED detection: {e}")
        status_text = "LED: Error"
        
    return led_state_this_frame, roi_color, status_text

def _process_frame_common(
    gray_frame, color_frame_to_annotate, camera_matrix, dist_coeffs, aruco_dict, aruco_parameters,
    marker_size_m, use_reference_marker_flag, reference_marker_id_val,
    positions_history,
    log_date_str, log_time_str, timestamp_for_plot,
    buoy_id_to_marker_ids_map_param,
    marker_buoy_geometry_config_param,
    buoy_cog_offsets_in_brf_config_param,
    log_all_marker_data_flag,
    plotter=None,
    ref_pose_filter=None,
    enable_led_detection_flag=False,
    led_detection_roi_tuple=None,
    led_off_baseline=0.0,
    led_brightness_threshold=30.0
):
    """
    Common processing logic for a single frame, used by both live and offline modes.
    Handles marker detection, pose estimation, buoy CoG calculation, logging, and drawing.

    :param gray_frame: Grayscale image frame.
    :param color_frame_to_annotate: Color image frame (will be modified with annotations).
    :param camera_matrix: Camera intrinsic matrix.
    :param dist_coeffs: Camera distortion coefficients.
    :param aruco_dict: ArUco dictionary object.
    :param aruco_parameters: ArUco detector parameters.
    :param marker_size_m: Physical size of markers in meters.
    :param use_reference_marker_flag: Boolean, if reference marker should be used.
    :param reference_marker_id_val: ID of the reference marker.
    :param positions_history: Dictionary to store buoy trajectory history.
    :param log_date_str: Date string for logging.
    :param log_time_str: Time string for logging.
    :param timestamp_for_plot: Numeric timestamp for plotting (e.g., seconds since start).
    :param buoy_id_to_marker_ids_map_param: Mapping of buoy IDs to their marker IDs.
    :param marker_buoy_geometry_config_param: Calibrated marker geometry for buoys.
    :param buoy_cog_offsets_in_brf_config_param: CoG offsets for buoys.
    :param log_all_marker_data_flag: Boolean, if individual marker data should be logged.
    :param plotter: RealTimePlotter instance or None.
    :param ref_pose_filter: PoseFilter instance for reference marker or None.
    :param enable_led_detection_flag: Boolean, if LED detection is enabled.
    :param led_detection_roi_tuple: Tuple (x,y,w,h) for LED ROI.
    :param led_off_baseline: Baseline brightness for LED OFF state.
    :param led_brightness_threshold: Threshold for LED ON state.

    :return: Tuple (annotated_frame, plot_image, cog_log_entries, individual_marker_log_entries,
                    positions_history, detected_buoys_summary, ref_marker_detected)
    """
    cog_log_entries = []
    individual_marker_log_entries = []
    plot_data_for_frame = {}
    
    annotated_frame = color_frame_to_annotate.copy() # Work on a copy to preserve original frame if needed

    # 1. LED Detection
    led_state_this_frame = False
    if enable_led_detection_flag:
        led_state_this_frame, _, _ = _perform_led_detection(
            annotated_frame, led_detection_roi_tuple, led_off_baseline, led_brightness_threshold
        )

    # 2. ArUco Marker Detection & Pose Estimation
    corners, ids, rvecs_all, tvecs_all, refined_corners_list = \
        _detect_and_estimate_markers(gray_frame, aruco_dict, aruco_parameters, marker_size_m, camera_matrix, dist_coeffs)

    ref_marker_detected = False
    ref_rvec, ref_tvec = None, None
    all_markers_data_this_frame = {}

    if ids is not None:
        # Draw detected markers on the frame
        aruco.drawDetectedMarkers(annotated_frame, refined_corners_list, ids)

        # 3. Process Reference Marker
        if use_reference_marker_flag:
            ref_marker_detected, ref_rvec, ref_tvec = _process_reference_marker(
                ids, rvecs_all, tvecs_all, refined_corners_list, camera_matrix, dist_coeffs, marker_size_m, ref_pose_filter, annotated_frame
            )
        
        # 4. Compile all marker data (camera and optionally relative poses)
        all_markers_data_this_frame = _get_all_marker_data(
            ids, rvecs_all, tvecs_all, refined_corners_list,
            use_reference_marker_flag, ref_marker_detected, ref_rvec, ref_tvec
        )

        # 5. Log individual marker data if enabled
        if log_all_marker_data_flag:
            _log_individual_marker_data(
                all_markers_data_this_frame, log_date_str, log_time_str, individual_marker_log_entries, use_reference_marker_flag
            )

        # 6. Estimate Buoy CoG Poses and Annotate
        detected_buoys_summary = _estimate_buoy_cog_poses(
            all_markers_data_this_frame, camera_matrix, dist_coeffs, marker_size_m,
            use_reference_marker_flag, ref_marker_detected, ref_rvec, ref_tvec,
            buoy_id_to_marker_ids_map_param, marker_buoy_geometry_config_param,
            buoy_cog_offsets_in_brf_config_param, log_date_str, log_time_str,
            led_state_this_frame, cog_log_entries, plot_data_for_frame,
            annotated_frame, positions_history
        )
    else:
        # No markers detected
        detected_buoys_summary = {}
        if ref_pose_filter:
            ref_pose_filter.reset() # Reset filter if reference marker is lost

    # 7. Update and Get Plot Image
    plot_image = None
    if plotter:
        plot_image = plotter.update_and_get_plot_image(plot_data_for_frame, timestamp_for_plot)

    # 8. Draw Trajectories
    _draw_trajectories(annotated_frame, positions_history)

    return (annotated_frame, plot_image, cog_log_entries, individual_marker_log_entries, positions_history, detected_buoys_summary, ref_marker_detected)

## END SECTION: CORE TRACKING LOGIC

## SECTION: APPLICATION MODES (LIVE & OFFLINE)

def track_buoy_with_aruco_6dof(parent_tk_window=None):
    """
    Starts a live video stream for real-time ArUco buoy tracking.
    Annotates frames, logs data, and optionally records video and displays plots.
    :param parent_tk_window: The parent Tkinter window for message boxes.
    """
    # Setup experiment directory for this live session
    global REVERSE_CAMERA_DISPLAY
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_live_{experiment_timestamp}"
    experiment_dir = os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created live experiment directory: {experiment_dir}")

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None) # cv2.CAP_DSHOW for better Windows camera compatibility
    if not cap.isOpened():
        messagebox.showerror("Camera Error", f"Cannot open camera at index {CAMERA_INDEX}. Please check camera connection and settings.", parent=parent_tk_window)
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    # Read actual camera properties (may differ from requested)
    actual_cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_cam_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {actual_cam_width}x{actual_cam_height} @ {actual_cam_fps:.1f} FPS")

    # Load camera calibration and ArUco dictionary
    cam_mat, dist_c = load_camera_calibration()
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    aruco_parameters = aruco.DetectorParameters() # Default parameters are usually fine

    # Initialize reference pose filter if enabled
    ref_pose_filter = PoseFilter(REF_FILTER_ALPHA_T, REF_FILTER_ALPHA_R) if ENABLE_REFERENCE_FILTER else None
    
    # Data structures for tracking and display
    positions_history = {} # Stores past center points for drawing trajectories
    info_panel_drawer = InfoPanelDrawer(panel_width=INFO_PANEL_WIDTH_PX)
    
    plotter = None
    output_frame_size = (actual_cam_width, actual_cam_height) # Size of the video frame output
    if SHOW_REALTIME_PLOTS:
        # Create plotter instance, ensuring it knows if relative pose is used
        plotter = RealTimePlotter(PLOT_HISTORY_LENGTH, PLOT_WIDTH_PX, actual_cam_height, list(BUOY_ID_TO_MARKER_IDS_MAP.keys()), USE_REFERENCE_MARKER)
        output_frame_size = (actual_cam_width + PLOT_WIDTH_PX, actual_cam_height) # Wider frame if plots are shown

    # Setup CSV logging files
    cog_csv_filename = os.path.join(experiment_dir, "tracking_data_cog.csv")
    # Define CSV header for CoG data, dynamically based on USE_REFERENCE_MARKER
    header_cog = ['Date', 'Time', 'Buoy_ID', 'LED_On']
    if USE_REFERENCE_MARKER:
        header_cog.extend(['Ref_Marker_Frame_Detected', 'CoG_Pos_X_Rel', 'CoG_Pos_Y_Rel', 'CoG_Pos_Z_Rel', 
                           'CoG_Rot_X_Rel_deg', 'CoG_Rot_Y_Rel_deg', 'CoG_Rot_Z_Rel_deg', 
                           'CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel'])
    header_cog.extend(['CoG_Pos_X_Cam', 'CoG_Pos_Y_Cam', 'CoG_Pos_Z_Cam', 
                       'CoG_Rot_X_Cam_deg', 'CoG_Rot_Y_Cam_deg', 'CoG_Rot_Z_Cam_deg', 
                       'CoG_Quat_X_Cam', 'CoG_Quat_Y_Cam', 'CoG_Quat_Z_Cam', 'CoG_Quat_W_Cam'])
    
    cog_csv_file = open(cog_csv_filename, 'w', newline='')
    csv_writer_cog = csv.writer(cog_csv_file)
    csv_writer_cog.writerow(header_cog)

    markers_csv_file = None
    if LOG_ALL_MARKER_DATA:
        markers_csv_filename = os.path.join(experiment_dir, "tracking_data_markers.csv")
        # Define CSV header for individual marker data
        header_markers = ['Log_Date', 'Log_Time', 'Marker_ID', 
                          'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z', 
                          'Cam_Rot_X_deg', 'Cam_Rot_Y_deg', 'Cam_Rot_Z_deg', 
                          'Cam_Quat_X', 'Cam_Quat_Y', 'Cam_Quat_Z', 'Cam_Quat_W']
        if USE_REFERENCE_MARKER:
            header_markers.extend(['Rel_Pos_X', 'Rel_Pos_Y', 'Rel_Pos_Z', 
                                   'Rel_Rot_X_deg', 'Rel_Rot_Y_deg', 'Rel_Rot_Z_deg', 
                                   'Rel_Quat_X', 'Rel_Quat_Y', 'Rel_Quat_Z', 'Rel_Quat_W'])
        markers_csv_file = open(markers_csv_filename, 'w', newline='')
        csv_writer_markers = csv.writer(markers_csv_file)
        csv_writer_markers.writerow(header_markers)

    is_recording = RECORD_VIDEO # Controlled by global setting, can be toggled by user
    recording_path = None
    vid_writer = None

    # Frame timing for FPS calculation
    frame_times = deque(maxlen=20) # Keep history of last 20 frame processing times

    try:
        while True:
            t_start_frame = time.perf_counter() # Start time for current frame processing

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera. Exiting live tracking.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get current timestamp for logging and plotting
            now_dt = datetime.now()
            date_string = now_dt.strftime("%Y-%m-%d")
            time_string = now_dt.strftime("%H:%M:%S.%f")[:-3] # Milliseconds up to 3 decimal places
            current_timestamp_seconds = time.time() # For plotter's relative time

            # Core frame processing function
            annotated_frame, plot_image, cog_logs, marker_logs, positions_history, buoy_summary, ref_detected = _process_frame_common(
                gray_frame, frame.copy(), cam_mat, dist_c, aruco_dict, aruco_parameters, MARKER_SIZE_METERS,
                USE_REFERENCE_MARKER, REFERENCE_MARKER_ID, positions_history,
                date_string, time_string, current_timestamp_seconds, BUOY_ID_TO_MARKER_IDS_MAP,
                MARKER_BUOY_GEOMETRY_CONFIG, BUOY_COG_OFFSETS_IN_BRF_CONFIG,
                LOG_ALL_MARKER_DATA, plotter, ref_pose_filter=ref_pose_filter,
                enable_led_detection_flag=ENABLE_LED_DETECTION,
                led_detection_roi_tuple=LED_DETECTION_ROI,
                led_off_baseline=LED_OFF_BASELINE_BRIGHTNESS,
                led_brightness_threshold=LED_BRIGHTNESS_THRESHOLD
            )
            
            # Write collected log data
            if cog_logs:
                csv_writer_cog.writerows(cog_logs)
            if marker_logs and markers_csv_file:
                csv_writer_markers.writerows(marker_logs)
            
            # Calculate and display processing FPS
            frame_times.append(time.perf_counter() - t_start_frame)
            proc_fps = 1.0 / np.mean(frame_times) if len(frame_times) > 1 else 0.0

            # Draw info panel
            info_panel_drawer.draw(annotated_frame, proc_fps, (actual_cam_width, actual_cam_height), actual_cam_fps, 
                                   (is_recording, recording_path), (USE_REFERENCE_MARKER, ref_detected, REFERENCE_MARKER_ID), 
                                   buoy_summary, is_live=True)

            # Combine video frame and plots if enabled
            final_display_frame = annotated_frame
            if plotter and plot_image is not None:
                # Create a blank frame to combine video and plots side-by-side
                final_display_frame = np.full((actual_cam_height, actual_cam_width + PLOT_WIDTH_PX, 3), 40, np.uint8) # Dark background
                final_display_frame[:, :actual_cam_width] = annotated_frame
                final_display_frame[:, actual_cam_width:] = cv2.resize(plot_image, (PLOT_WIDTH_PX, actual_cam_height)) # Resize plot to match video height

            # Add general controls overlay
            cv2.putText(final_display_frame, "Q: Quit | R: Reverse Disp | V: Record", (20, actual_cam_height - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0,165,255), OVERLAY_THICKNESS)
            
            # Reverse display if enabled
            display_for_screen = cv2.flip(final_display_frame, 1) if REVERSE_CAMERA_DISPLAY else final_display_frame
            
            # Show frame
            cv2.imshow('Buoy 6DOF Tracking (Live)', display_for_screen)

            # Handle video recording
            if is_recording:
                if vid_writer is None:
                    # Initialize video writer if recording just started
                    recording_path = os.path.join(experiment_dir, f"live_annotated_{experiment_timestamp}{VIDEO_EXTENSION}")
                    vid_writer = get_video_writer(output_frame_size, recording_path, fps=actual_cam_fps)
                    if not vid_writer: # If writer creation failed
                        is_recording = False
                        print("ERROR: Could not start video recording.")
                    else:
                        print(f"Recording started to {recording_path}")
                if vid_writer:
                    vid_writer.write(final_display_frame) # Write the combined frame

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Quit
                break
            elif key == ord('r'): # Toggle reverse display
                REVERSE_CAMERA_DISPLAY = not REVERSE_CAMERA_DISPLAY
            elif key == ord('v'): # Toggle recording
                is_recording = not is_recording
                if not is_recording and vid_writer:
                    # Stop recording
                    vid_writer.release()
                    vid_writer = None
                    print(f"Recording stopped. Video saved to {recording_path}")
                    recording_path = None # Clear path once stopped

    finally:
        # Clean up resources
        if vid_writer:
            vid_writer.release()
        if cog_csv_file:
            cog_csv_file.close()
        if markers_csv_file:
            markers_csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Live Tracking Ended", f"Experiment data saved in:\n{experiment_dir}", parent=parent_tk_window)

def process_video_offline(input_video_path, use_ref_marker_override,
                          gui_status_label=None, gui_progress_bar=None,
                          video_start_date_str=None, video_start_time_str=None,
                          led_roi_override=None, led_baseline_override=None):
    """
    Processes a pre-recorded video file for ArUco buoy tracking offline.
    Generates annotated video and CSV log files.
    :param input_video_path: Path to the input video file.
    :param use_ref_marker_override: Boolean, whether to use reference marker for this processing run.
    :param gui_status_label: Tkinter Label to update status.
    :param gui_progress_bar: Tkinter Progressbar to update progress.
    :param video_start_date_str: Optional date string for absolute timestamps (YYYY-MM-DD).
    :param video_start_time_str: Optional time string for absolute timestamps (HH:MM:SS).
    :param led_roi_override: Optional override for LED detection ROI (x,y,w,h).
    :param led_baseline_override: Optional override for LED off-state baseline brightness.
    """
    parent_tk = gui_status_label.master if gui_status_label else None

    if not os.path.exists(input_video_path):
        messagebox.showerror("File Error", f"Input video file not found: {input_video_path}", parent=parent_tk)
        return

    # Setup experiment directory for this offline session
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    experiment_name = f"experiment_offline_{video_base_name}_{experiment_timestamp}"
    experiment_dir = os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created offline experiment directory: {experiment_dir}")

    # Copy original video to experiment folder for record-keeping
    try:
        shutil.copy2(input_video_path, os.path.join(experiment_dir, f"original_{os.path.basename(input_video_path)}"))
        print(f"Copied input video to {experiment_dir}")
    except Exception as e:
        print(f"Warning: Could not copy input video to experiment folder: {e}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        messagebox.showerror("Video Error", f"Cannot open video file: {input_video_path}", parent=parent_tk)
        return

    # Get video properties
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if gui_progress_bar:
        gui_progress_bar["maximum"] = total_frames if total_frames > 0 else 100 # Avoid division by zero

    print(f"Processing video: {input_video_path} ({vid_width}x{vid_height} @ {vid_fps:.1f} FPS, {total_frames} frames)")

    # Load camera calibration and ArUco dictionary
    cam_mat, dist_c = load_camera_calibration()
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    aruco_parameters = aruco.DetectorParameters()

    # Initialize reference pose filter (if enabled, using global setting)
    ref_pose_filter = PoseFilter(REF_FILTER_ALPHA_T, REF_FILTER_ALPHA_R) if ENABLE_REFERENCE_FILTER else None

    # Data structures for tracking and display
    positions_history = {} # Stores past center points for drawing trajectories
    info_panel_drawer = InfoPanelDrawer(panel_width=INFO_PANEL_WIDTH_PX)
    
    plotter = None
    output_frame_size = (vid_width, vid_height)
    if SHOW_REALTIME_PLOTS:
        plotter = RealTimePlotter(PLOT_HISTORY_LENGTH, PLOT_WIDTH_PX, vid_height, list(BUOY_ID_TO_MARKER_IDS_MAP.keys()), use_ref_marker_override)
        output_frame_size = (vid_width + PLOT_WIDTH_PX, vid_height) # Wider frame if plots are shown

    # Setup annotated video writer
    annotated_video_path = os.path.join(experiment_dir, f"annotated_{experiment_timestamp}{VIDEO_EXTENSION}")
    vid_writer = get_video_writer(output_frame_size, annotated_video_path, fps=vid_fps)
    if not vid_writer:
        messagebox.showerror("Video Writer Error", "Failed to create annotated video file. Processing will continue, but no video will be saved.", parent=parent_tk)

    # Setup CSV logging files
    cog_csv_filename = os.path.join(experiment_dir, "tracking_data_cog.csv")
    header_cog = ['Date', 'Time', 'Buoy_ID', 'LED_On']
    if use_ref_marker_override: # Header adjusts based on runtime override, not global config
        header_cog.extend(['Ref_Marker_Frame_Detected', 'CoG_Pos_X_Rel', 'CoG_Pos_Y_Rel', 'CoG_Pos_Z_Rel', 
                           'CoG_Rot_X_Rel_deg', 'CoG_Rot_Y_Rel_deg', 'CoG_Rot_Z_Rel_deg', 
                           'CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel'])
    header_cog.extend(['CoG_Pos_X_Cam', 'CoG_Pos_Y_Cam', 'CoG_Pos_Z_Cam', 
                       'CoG_Rot_X_Cam_deg', 'CoG_Rot_Y_Cam_deg', 'CoG_Rot_Z_Cam_deg', 
                       'CoG_Quat_X_Cam', 'CoG_Quat_Y_Cam', 'CoG_Quat_Z_Cam', 'CoG_Quat_W_Cam'])
    
    cog_csv_file = open(cog_csv_filename, 'w', newline='')
    csv_writer_cog = csv.writer(cog_csv_file)
    csv_writer_cog.writerow(header_cog)

    markers_csv_file = None
    if LOG_ALL_MARKER_DATA:
        markers_csv_filename = os.path.join(experiment_dir, "tracking_data_markers.csv")
        header_markers = ['Log_Date', 'Log_Time', 'Marker_ID', 
                          'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z', 
                          'Cam_Rot_X_deg', 'Cam_Rot_Y_deg', 'Cam_Rot_Z_deg', 
                          'Cam_Quat_X', 'Cam_Quat_Y', 'Cam_Quat_Z', 'Cam_Quat_W']
        if use_ref_marker_override:
            header_markers.extend(['Rel_Pos_X', 'Rel_Pos_Y', 'Rel_Pos_Z', 
                                   'Rel_Rot_X_deg', 'Rel_Rot_Y_deg', 'Rel_Rot_Z_deg', 
                                   'Rel_Quat_X', 'Rel_Quat_Y', 'Rel_Quat_Z', 'Rel_Quat_W'])
        markers_csv_file = open(markers_csv_filename, 'w', newline='')
        csv_writer_markers = csv.writer(markers_csv_file)
        csv_writer_markers.writerow(header_markers)
    
    # Determine timestamping method
    base_timestamp = None
    timestamp_note = "Using relative video time."
    if video_start_date_str and video_start_time_str:
        try:
            base_timestamp = datetime.strptime(f"{video_start_date_str.strip()} {video_start_time_str.strip()}", "%Y-%m-%d %H:%M:%S")
            timestamp_note = "Using user-provided start timestamp."
            print(f"Video timestamping will be absolute, starting from: {base_timestamp}")
        except ValueError:
            messagebox.showwarning("Date/Time Parse Error", "Could not parse provided date/time. Using relative video time for timestamps.", parent=parent_tk)
            timestamp_note = "Fallback to relative video time due to parse error."
            print("Fallback to relative video time for timestamps.")

    frame_count = 0
    start_processing_time = time.time() # To calculate overall processing FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break
            
            frame_count += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get current time in video (milliseconds from start)
            frame_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_sec_for_plot = frame_msec / 1000.0 # For plotting X-axis

            # Generate date/time strings for logging
            if base_timestamp:
                current_dt = base_timestamp + timedelta(milliseconds=frame_msec)
                date_string = current_dt.strftime("%Y-%m-%d")
                time_string = current_dt.strftime("%H:%M:%S.%f")[:-3]
            else:
                # If no base timestamp, use a placeholder date and time from video start
                date_string = datetime.now().strftime("%Y-%m-%d") # Use current system date
                time_string = (datetime.min + timedelta(milliseconds=frame_msec)).strftime("%H:%M:%S.%f")[:-3]

            # Core frame processing
            annotated_frame, plot_image, cog_logs, marker_logs, positions_history, buoy_summary, ref_detected = _process_frame_common(
                gray_frame, frame.copy(), cam_mat, dist_c, aruco_dict, aruco_parameters, MARKER_SIZE_METERS,
                use_ref_marker_override, REFERENCE_MARKER_ID, positions_history,
                date_string, time_string, frame_sec_for_plot, BUOY_ID_TO_MARKER_IDS_MAP,
                MARKER_BUOY_GEOMETRY_CONFIG, BUOY_COG_OFFSETS_IN_BRF_CONFIG,
                LOG_ALL_MARKER_DATA, plotter, ref_pose_filter=ref_pose_filter,
                enable_led_detection_flag=ENABLE_LED_DETECTION,
                led_detection_roi_tuple=led_roi_override,
                led_off_baseline=led_baseline_override if led_baseline_override is not None else LED_OFF_BASELINE_BRIGHTNESS,
                led_brightness_threshold=LED_BRIGHTNESS_THRESHOLD
            )
            
            # Write log data
            if cog_logs:
                csv_writer_cog.writerows(cog_logs)
            if marker_logs and markers_csv_file:
                csv_writer_markers.writerows(marker_logs)
            
            # Calculate overall processing FPS
            proc_fps = frame_count / (time.time() - start_processing_time) if (time.time() - start_processing_time) > 0 else 0.0

            # Draw info panel (status for offline processing is fixed for recording/reference)
            info_panel_drawer.draw(annotated_frame, proc_fps, (vid_width, vid_height), vid_fps, 
                                   (vid_writer is not None, annotated_video_path if vid_writer else "N/A"), 
                                   (use_ref_marker_override, ref_detected, REFERENCE_MARKER_ID), 
                                   buoy_summary, is_live=False)

            # Combine video frame and plots if enabled
            final_display_frame = annotated_frame
            if plotter and plot_image is not None:
                final_display_frame = np.full((vid_height, vid_width + PLOT_WIDTH_PX, 3), 40, np.uint8)
                final_display_frame[:, :vid_width] = annotated_frame
                final_display_frame[:, vid_width:] = cv2.resize(plot_image, (PLOT_WIDTH_PX, vid_height))

            # Add progress text to the display frame
            progress_text = f"Frame: {frame_count}/{total_frames}"
            (text_width, text_height), _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OVERLAY_THICKNESS)
            cv2.putText(final_display_frame, progress_text, (vid_width - text_width - 20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OFFLINE_PROGRESS_COLOR, OVERLAY_THICKNESS)

            # Write annotated frame to output video
            if vid_writer:
                vid_writer.write(final_display_frame)

            # Update GUI progress bar and status label periodically
            if gui_progress_bar and (frame_count % 25 == 0 or frame_count == total_frames):
                if total_frames > 0:
                    progress_pct = (frame_count / total_frames) * 100
                    gui_progress_bar["value"] = frame_count
                    if gui_status_label:
                        gui_status_label.config(text=f"Processing... {progress_pct:.1f}%")
                    if parent_tk:
                        parent_tk.update_idletasks() # Update GUI to show progress

            # Show live preview of offline processing (optional)
            if SHOW_OFFLINE_PROCESSING_PREVIEW:
                preview_width = int(output_frame_size[0] * OFFLINE_PREVIEW_RESIZE_FACTOR)
                preview_height = int(output_frame_size[1] * OFFLINE_PREVIEW_RESIZE_FACTOR)
                cv2.imshow('Offline Processing Preview', cv2.resize(final_display_frame, (preview_width, preview_height)))
                if (cv2.waitKey(1) & 0xFF) == ord('q'): # Allow quitting preview
                    print("Offline processing preview manually stopped.")
                    break

    finally:
        # Clean up resources
        if vid_writer:
            vid_writer.release()
        if cog_csv_file:
            cog_csv_file.close()
        if markers_csv_file:
            markers_csv_file.close()
        cap.release()
        if SHOW_OFFLINE_PROCESSING_PREVIEW:
            cv2.destroyAllWindows()

        saved_files_message = f"CoG data saved to: {cog_csv_filename}"
        if LOG_ALL_MARKER_DATA:
            saved_files_message += f"\nIndividual markers data saved to: {markers_csv_filename}"
        if vid_writer: # Only if video writer was successfully created
            saved_files_message += f"\nAnnotated video saved to: {annotated_video_path}"

        final_message = (f"Offline processing complete for '{os.path.basename(input_video_path)}'.\n"
                         f"All output data saved in experiment folder:\n{experiment_dir}\n\n"
                         f"{saved_files_message}\n\n"
                         f"Timestamping note: {timestamp_note}")
        
        if gui_status_label:
            gui_status_label.config(text=f"Processing Done. Data in: {experiment_name}")
        messagebox.showinfo("Processing Complete", final_message, parent=parent_tk)

## END SECTION: APPLICATION MODES (LIVE & OFFLINE)

## SECTION: CALIBRATION TOOLS

def calibrate_camera(parent_tk_window=None):
    """
    Performs camera intrinsic calibration using a chessboard pattern.
    Guides the user through capturing images and then calculates calibration parameters.
    :param parent_tk_window: The parent Tkinter window for message boxes.
    """
    print("Starting Camera Calibration process...")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", f"Cannot open camera {CAMERA_INDEX} for calibration. Check connection.", parent=parent_tk_window)
        return

    # Set camera resolution (best effort)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    
    # Get actual resolution (might differ)
    actual_cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened for calibration: {actual_cam_width}x{actual_cam_height}")

    # Chessboard setup
    chessboard_shape = CALIBRATION_CHESSBOARD_SHAPE # e.g., (9, 6) internal corners
    square_size_meters = CHESSBOARD_SQUARE_SIZE_MM / 1000.0 # Convert mm to meters

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2) * square_size_meters

    obj_points_list = [] # 3D points in real world space
    img_points_list = [] # 2D points in image plane
    gray_frame_shape = None # To store the image size for calibration

    window_name = 'Camera Calibration - Q to Finish & Calibrate'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allow resizing

    rev_disp_calib = REVERSE_CAMERA_DISPLAY # Start with current setting

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot grab frame for calibration. Camera disconnected or error.")
            break

        gray_frame_calib = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if gray_frame_shape is None:
            gray_frame_shape = gray_frame_calib.shape[::-1] # (width, height)

        display_frame_calib = frame.copy()

        # Find the chessboard corners
        ret_corners, corners_found = cv2.findChessboardCorners(gray_frame_calib, chessboard_shape, None)

        text_color = (0,0,255) # Red by default
        status_text = "Aim at chessboard..."

        if ret_corners:
            text_color = (0,255,0) # Green if detected
            status_text = "Chessboard detected! Press 'c' to capture."
            # Draw corners on the display frame
            cv2.drawChessboardCorners(display_frame_calib, chessboard_shape, corners_found, ret_corners)

        # Overlay instructions and status
        cv2.putText(display_frame_calib, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, text_color, OVERLAY_THICKNESS)
        cv2.putText(display_frame_calib, f"Captured images: {len(obj_points_list)}", (20, actual_cam_height - 40), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 0, 0), OVERLAY_THICKNESS)
        cv2.putText(display_frame_calib, f"Reverse Display: {'ON' if rev_disp_calib else 'OFF'} (Press 'r')", (20, actual_cam_height - 100), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.7, (0, 165, 255), OVERLAY_THICKNESS)

        # Apply display flip if enabled
        if rev_disp_calib:
            display_frame_calib = cv2.flip(display_frame_calib, 1)

        cv2.imshow(window_name, display_frame_calib)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Quit and attempt calibration
            break
        elif key == ord('r'): # Toggle reverse display
            rev_disp_calib = not rev_disp_calib
        elif key == ord('c') and ret_corners: # Capture image if chessboard is found
            criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray_frame_calib, corners_found, (11,11), (-1,-1), criteria_subpix)
            
            obj_points_list.append(objp)
            img_points_list.append(corners_refined)
            print(f"Calibration image captured ({len(obj_points_list)} total).")
            
            # Briefly show confirmation
            cv2.drawChessboardCorners(frame, chessboard_shape, corners_refined, True)
            temp_display_confirm = frame.copy()
            if rev_disp_calib: temp_display_confirm = cv2.flip(temp_display_confirm, 1)
            cv2.imshow(window_name, temp_display_confirm)
            cv2.waitKey(500) # Show for 0.5 seconds

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points_list) > 3 and gray_frame_shape is not None:
        print(f"Calculating camera calibration with {len(obj_points_list)} images...")
        try:
            ret_calib, camera_matrix_calib, dist_coeffs_calib, rvecs_calib, tvecs_calib = \
                cv2.calibrateCamera(obj_points_list, img_points_list, gray_frame_shape, None, None)

            if ret_calib:
                # Calculate mean reprojection error
                mean_error = 0
                for i in range(len(obj_points_list)):
                    img_points_reprojected, _ = cv2.projectPoints(obj_points_list[i], rvecs_calib[i], tvecs_calib[i], camera_matrix_calib, dist_coeffs_calib)
                    mean_error += cv2.norm(img_points_list[i], img_points_reprojected, cv2.NORM_L2) / len(img_points_reprojected)
                mean_error /= len(obj_points_list)

                reprojection_error_message = f"Mean Reprojection Error: {mean_error:.4f} pixels"
                
                print(f"Camera Matrix:\n{camera_matrix_calib}\nDistortion Coefficients:\n{dist_coeffs_calib.ravel()}\n{reprojection_error_message}")

                # Save calibration
                calibration_filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME, CALIBRATION_FILE)
                np.savez(calibration_filepath, camera_matrix=camera_matrix_calib, dist_coeffs=dist_coeffs_calib, 
                         reprojection_error=mean_error, num_images=len(obj_points_list))
                
                messagebox.showinfo("Calibration Complete", 
                                    f"Camera calibration successful and saved to:\n'{calibration_filepath}'.\n\n"
                                    f"{reprojection_error_message}", parent=parent_tk_window)
            else:
                messagebox.showwarning("Calibration Failed", "cv2.calibrateCamera returned false. Calibration could not be computed.", parent=parent_tk_window)
        except Exception as e:
            messagebox.showerror("Calibration Error", f"An error occurred during calibration calculation: {e}", parent=parent_tk_window)
    else:
        messagebox.showwarning("Calibration Needed", 
                               f"Insufficient images captured for calibration. Need at least 4 valid chessboard views (got {len(obj_points_list)}).", 
                               parent=parent_tk_window)

def calibrate_buoy_marker_geometry(buoy_id_to_calibrate, parent_tk_window=None, num_frames_to_capture=30):
    """
    Calibrates the relative 6DOF poses of markers on a specific buoy with respect to its "X0" marker.
    This creates a fixed geometry for the buoy's body reference frame.
    :param buoy_id_to_calibrate: The ID of the buoy whose markers are to be calibrated.
    :param parent_tk_window: The parent Tkinter window for message boxes.
    :param num_frames_to_capture: Target number of good frames to capture for averaging.
    """
    global ALL_BUOY_GEOMETRY_CALIBRATIONS

    if buoy_id_to_calibrate not in BUOY_ID_TO_MARKER_IDS_MAP:
        messagebox.showerror("Configuration Error", f"Buoy ID {buoy_id_to_calibrate} is not defined in the configuration (BUOY_ID_TO_MARKER_IDS_MAP).", parent=parent_tk_window)
        return

    target_marker_ids_on_buoy = BUOY_ID_TO_MARKER_IDS_MAP[buoy_id_to_calibrate]
    
    if len(target_marker_ids_on_buoy) < 2:
        messagebox.showinfo("Info", f"Buoy ID {buoy_id_to_calibrate} has fewer than 2 markers defined. No relative geometry calibration is needed or possible.", parent=parent_tk_window)
        return

    id_X0 = target_marker_ids_on_buoy[0] # Assumes the first marker in the list is the reference (X0)
    ids_to_calibrate_relative_to_X0 = target_marker_ids_on_buoy[1:] # All other markers on the buoy

    msg = (f"Starting relative geometry calibration for Buoy ID: {buoy_id_to_calibrate}.\n\n"
           f"The marker designated as 'X0' (origin of buoy frame) is ID: {id_X0}\n"
           f"Markers to calibrate relative to X0 are: {ids_to_calibrate_relative_to_X0}\n\n"
           f"Instructions:\n"
           f"1. Position the buoy so that the X0 marker AND at least one other target marker ({ids_to_calibrate_relative_to_X0}) are clearly visible to the camera.\n"
           f"2. Move the buoy around slightly to get multiple perspective views (do not obscure markers).\n"
           f"3. Press 'c' to capture a good frame. You need {num_frames_to_capture} total.\n"
           f"4. Press 'q' to finish and calculate the calibration at any time.")
    messagebox.showinfo("Buoy Geometry Calibration Setup", msg, parent=parent_tk_window)

    cam_matrix, dist_coeffs = load_camera_calibration()
    aruco_dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters_obj = aruco.DetectorParameters()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", f"Cannot open camera {CAMERA_INDEX} for buoy calibration.", parent=parent_tk_window)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

    # Store collected relative poses for each marker (list of tvecs and rvecs)
    collected_relative_poses = {marker_id: {'tvecs': [], 'rvecs': []} for marker_id in ids_to_calibrate_relative_to_X0}
    frames_captured_count = 0

    window_name = f"Buoy {buoy_id_to_calibrate} Geometry Calibration - Press 'c' to capture, 'q' to finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while frames_captured_count < num_frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Failed to grab frame during buoy calibration. Check camera connection.", parent=parent_tk_window)
            break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict_obj, parameters=parameters_obj)
        
        poses_cam_this_frame = {} # Store (rvec, tvec) for markers detected in current frame

        if ids is not None and len(ids) > 0:
            # Draw detected markers on the display frame
            aruco.drawDetectedMarkers(display_frame, corners, ids)

            # Estimate poses for all detected markers
            rvecs_est, tvecs_est, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_METERS, cam_matrix, dist_coeffs)

            for i, marker_id_arr in enumerate(ids):
                marker_id = marker_id_arr[0]
                if marker_id in target_marker_ids_on_buoy:
                    # Store detected pose for potential use in relative calculation
                    poses_cam_this_frame[marker_id] = (rvecs_est[i].reshape(3,1), tvecs_est[i].reshape(3,1))
                    # Draw axes for visual confirmation
                    cv2.drawFrameAxes(display_frame, cam_matrix, dist_coeffs, rvecs_est[i], tvecs_est[i], MARKER_SIZE_METERS, AXES_THICKNESS//2)
        
        # Check visibility status of required markers
        x0_is_visible = id_X0 in poses_cam_this_frame
        any_xi_visible_with_x0 = any(xi_id in poses_cam_this_frame for xi_id in ids_to_calibrate_relative_to_X0)

        status_text_calib = ""
        status_color_calib = (0, 165, 255) # Orange-Blue default

        if x0_is_visible and any_xi_visible_with_x0:
            status_text_calib = "STATUS: OK! Press 'c' to capture this view."
            status_color_calib = (0, 255, 0) # Green
        elif x0_is_visible:
            status_text_calib = f"STATUS: X0 (ID {id_X0}) detected. Show another marker with X0."
            status_color_calib = (0, 255, 255) # Yellow
        else:
            status_text_calib = f"STATUS: X0 (ID {id_X0}) NOT visible. Please ensure it's in view."
            status_color_calib = (0, 0, 255) # Red

        # Overlay status and instructions
        cv2.putText(display_frame, f"Buoy ID: {buoy_id_to_calibrate} (X0 Marker ID: {id_X0})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (255, 255, 0), OVERLAY_THICKNESS)
        cv2.putText(display_frame, status_text_calib, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.9, status_color_calib, OVERLAY_THICKNESS)
        cv2.putText(display_frame, f"Captured valid views: {frames_captured_count}/{num_frames_to_capture}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, (0, 255, 255), OVERLAY_THICKNESS)
        
        # Display sample counts per marker
        y_offset_counts = 200
        for xi_id_disp in ids_to_calibrate_relative_to_X0:
            count_xi = len(collected_relative_poses[xi_id_disp]['tvecs'])
            cv2.putText(display_frame, f"  > Marker {xi_id_disp} samples: {count_xi}", (20, y_offset_counts), cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.8, (200, 200, 50), OVERLAY_THICKNESS)
            y_offset_counts += 40

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Quit calibration process
            break
        elif key == ord('c'): # Capture frame for calibration
            if not x0_is_visible:
                messagebox.showwarning("Capture Skipped", f"Cannot capture: Reference marker X0 (ID {id_X0}) is not visible.", parent=parent_tk_window)
                continue

            rvec_X0_cam, tvec_X0_cam = poses_cam_this_frame[id_X0]
            captured_pair_this_press = False

            # Calculate and store relative poses for all other visible buoy markers
            for marker_id_Xi in ids_to_calibrate_relative_to_X0:
                if marker_id_Xi in poses_cam_this_frame:
                    try:
                        rvec_Xi_cam, tvec_Xi_cam = poses_cam_this_frame[marker_id_Xi]
                        # Transform pose of Xi marker to be relative to X0 marker
                        rvec_X0_Xi, tvec_X0_Xi = transform_pose_to_reference(rvec_Xi_cam, tvec_Xi_cam, rvec_X0_cam, tvec_X0_cam)
                        
                        collected_relative_poses[marker_id_Xi]['tvecs'].append(tvec_X0_Xi.flatten())
                        collected_relative_poses[marker_id_Xi]['rvecs'].append(rvec_X0_Xi.flatten())
                        print(f"  Captured relative pose for X0 <-- Marker ID {marker_id_Xi}")
                        captured_pair_this_press = True
                    except Exception as e:
                        print(f"Error calculating relative pose for marker {marker_id_Xi}: {e}")
            
            if captured_pair_this_press:
                frames_captured_count += 1
                # Briefly show success
                cv2.putText(display_frame, "CAPTURED!", (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(300) # Show "CAPTURED!" message for 0.3 seconds
            else:
                messagebox.showinfo("Capture Info", "X0 was visible, but no other target markers were visible in this frame to form a pair. Try a different view.", parent=parent_tk_window)

    cap.release()
    cv2.destroyAllWindows()

    # Check if any valid data was captured
    if not any(len(data['tvecs']) > 0 for data in collected_relative_poses.values()):
        messagebox.showwarning("Incomplete Calibration", "No valid data pairs (X0, Xi) were captured for any marker on this buoy. Calibration cannot be performed.", parent=parent_tk_window)
        return

    # Check for markers with insufficient samples
    low_sample_markers = [mid for mid, data in collected_relative_poses.items() if 0 < len(data['tvecs']) < BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR]
    if low_sample_markers:
        if not messagebox.askyesno("Warning: Low Samples", 
                                   f"The following markers have fewer than {BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR} samples:\n{low_sample_markers}\n"
                                   "This may lead to less accurate calibration for these markers. Do you want to proceed anyway?", 
                                   icon='warning', parent=parent_tk_window):
            return # User decided to cancel

    print(f"\nCalculating averaged poses for Buoy ID {buoy_id_to_calibrate}...")
    buoy_calib_results = {}

    # Calculate average poses for each marker Xi relative to X0
    for marker_id_Xi, data in collected_relative_poses.items():
        if not data['tvecs']:
            # If no data was collected for this marker, store None
            buoy_calib_results[str(marker_id_Xi)] = {'t_rel_to_X0': None, 'rvec_rel_to_X0': None}
            print(f"  Marker {marker_id_Xi}: No data captured, leaving as uncalibrated.")
            continue

        # Average translations
        avg_tvec = np.mean(np.array(data['tvecs']), axis=0)

        # Average rotations using quaternion averaging
        quats = [R.from_rotvec(rvec).as_quat() for rvec in data['rvecs']]
        avg_quat = average_quaternions_weighted(quats, [1.0] * len(quats)) # Equal weights
        avg_rvec = R.from_quat(avg_quat).as_rotvec()

        print(f"  Marker {marker_id_Xi} (Samples: {len(data['tvecs'])}):")
        print(f"    Avg Translation (X0 to Xi): {avg_tvec}")
        print(f"    Avg Rotation (X0 to Xi): {np.rad2deg(avg_rvec)} deg (Rodrigues)")
        
        # Store results, ensuring they are float32 NumPy arrays
        buoy_calib_results[str(marker_id_Xi)] = {
            't_rel_to_X0': avg_tvec.astype(np.float32),
            'rvec_rel_to_X0': avg_rvec.astype(np.float32)
        }
    
    # Update the global calibration dictionary
    ALL_BUOY_GEOMETRY_CALIBRATIONS[str(buoy_id_to_calibrate)] = buoy_calib_results
    
    # Save all buoy geometry calibrations to file
    save_buoy_geometry_calibrations(ALL_BUOY_GEOMETRY_CALIBRATIONS)
    
    # Re-initialize main buoy configurations to reflect new calibration
    initialize_buoy_configs()

    messagebox.showinfo("Calibration Complete", 
                        f"Geometry calibration for Buoy ID {buoy_id_to_calibrate} finished and saved. "
                        "The updated geometry will be used in subsequent tracking sessions.", 
                        parent=parent_tk_window)

## END SECTION: CALIBRATION TOOLS

## SECTION: PATTERN GENERATION TOOLS

def generate_aruco_marker(marker_id, size_px=600, border_px=50, save=True, parent_tk_window=None):
    """
    Generates a single ArUco marker image and optionally saves it.
    Adds a border, ID text, and coordinate axes.
    :param marker_id: The ID of the ArUco marker to generate.
    :param size_px: The size of the marker (excluding border) in pixels.
    :param border_px: The width of the white border in pixels.
    :param save: If True, saves the generated image to file.
    :param parent_tk_window: Parent Tkinter window for message boxes.
    :return: The generated image as a NumPy array (BGR format).
    """
    aruco_dict_gen = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    
    try:
        # Generate the basic ArUco marker (grayscale)
        img_gray = aruco.generateImageMarker(aruco_dict_gen, marker_id, size_px)
    except cv2.error as e:
        messagebox.showerror("Marker Generation Error", f"Failed to generate marker ID {marker_id}: {e}", parent=parent_tk_window)
        return None
    
    # Pad with white border and convert to BGR
    img_bgr_bordered = cv2.cvtColor(np.pad(img_gray, border_px, 'constant', constant_values=255), cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0,0,0) # Black
    total_size = size_px + 2 * border_px

    # Add marker ID text to the bottom right
    id_text = f"ID: {marker_id}"
    (text_width, text_height), _ = cv2.getTextSize(id_text, font, 0.7, 2)
    cv2.putText(img_bgr_bordered, id_text, 
                (total_size - text_width - max(5, int(border_px * 0.1)), total_size - max(5, int(border_px * 0.1))), 
                font, 0.7, text_color, 2)

    # Add simple coordinate axes (if border is large enough)
    if border_px >= 20:
        axis_thickness = 2
        arrow_length = max(10, int(border_px * 0.6))
        tip_length = 0.3
        label_font_scale = 0.5
        label_font_thickness = 1
        label_padding = max(3, int(border_px * 0.1))

        # X-axis (Red)
        origin_x = (border_px, border_px + size_px + border_px // 2) # X-axis points right
        cv2.arrowedLine(img_bgr_bordered, origin_x, (origin_x[0] + arrow_length, origin_x[1]), (0, 0, 255), axis_thickness, tipLength=tip_length)
        (xlw, xlh), _ = cv2.getTextSize("X", font, label_font_scale, label_font_thickness)
        cv2.putText(img_bgr_bordered, "X", (origin_x[0] + arrow_length + label_padding, origin_x[1] + xlh // 2), font, label_font_scale, (0, 0, 255), label_font_thickness)

        # Y-axis (Green)
        origin_y = (border_px - border_px // 2, border_px + size_px) # Y-axis points up
        cv2.arrowedLine(img_bgr_bordered, origin_y, (origin_y[0], origin_y[1] - arrow_length), (0, 255, 0), axis_thickness, tipLength=tip_length)
        (ylw, ylh), _ = cv2.getTextSize("Y", font, label_font_scale, label_font_thickness)
        cv2.putText(img_bgr_bordered, "Y", (origin_y[0] - ylw // 2, origin_y[1] - arrow_length - label_padding), font, label_font_scale, (0, 255, 0), label_font_thickness)
        
        # Z-axis (Blue, out of plane) - represented as a circle with dot
        z_center = (border_px // 2, border_px // 2)
        z_radius = max(5, border_px // 4)
        cv2.circle(img_bgr_bordered, z_center, z_radius, (255, 0, 0), axis_thickness) # Circle for Z
        cv2.circle(img_bgr_bordered, z_center, max(1, z_radius // 3), (255, 0, 0), -1) # Dot for Z (out of plane)
        (zlw, zlh), _ = cv2.getTextSize("Z", font, label_font_scale, label_font_thickness)
        cv2.putText(img_bgr_bordered, "Z", (z_center[0] + z_radius + label_padding, z_center[1] + zlh // 2), font, label_font_scale, (255, 0, 0), label_font_thickness)
        (zoutw, zouth), _ = cv2.getTextSize("(out)", font, label_font_scale * 0.8, label_font_thickness)
        cv2.putText(img_bgr_bordered, "(out)", (z_center[0] + z_radius + label_padding, z_center[1] + zlh // 2 + zouth + 2), font, label_font_scale * 0.8, (255, 0, 0), label_font_thickness)


    # Add "REFERENCE MARKER" text if applicable
    if marker_id == REFERENCE_MARKER_ID and USE_REFERENCE_MARKER:
        ref_text = "REFERENCE MARKER"
        (trw, trh), _ = cv2.getTextSize(ref_text, font, 0.8, 2)
        cv2.putText(img_bgr_bordered, ref_text, ((total_size - trw) // 2, border_px // 2 + trh // 2 if border_px > trh else trh + 5), font, 0.8, text_color, 2)

    if save:
        filename_suffix = f"_ID{marker_id}" + ('_REFERENCE' if marker_id == REFERENCE_MARKER_ID and USE_REFERENCE_MARKER else '')
        filepath = os.path.join(DATA_OUTPUT_DIRECTORY, PATTERNS_DIRECTORY_NAME, f"aruco_marker{filename_suffix}.png")
        try:
            cv2.imwrite(filepath, img_bgr_bordered)
            msg = f"ArUco marker saved to: {filepath}"
        except Exception as e:
            msg = f"Error saving ArUco marker to {filepath}: {e}"
        
        if parent_tk_window:
            messagebox.showinfo("Marker Generation", msg, parent=parent_tk_window)
        else:
            print(msg)
    
    return img_bgr_bordered

def create_chessboard(squares_x, squares_y, square_size_px=100, save=True, parent_tk_window=None):
    """
    Generates a chessboard pattern image for camera calibration and optionally saves it.
    Adds instructional text.
    :param squares_x: Number of internal corners horizontally.
    :param squares_y: Number of internal corners vertically.
    :param square_size_px: Size of each square in pixels.
    :param save: If True, saves the generated image to file.
    :param parent_tk_window: Parent Tkinter window for message boxes.
    :return: The generated image as a NumPy array (grayscale).
    """
    # Calculate dimensions including one extra row/column for corners
    pattern_cols = squares_x + 1
    pattern_rows = squares_y + 1
    
    img_width = pattern_cols * square_size_px
    img_height = pattern_rows * square_size_px

    board = np.full((img_height, img_width), 255, dtype=np.uint8) # Start with a white board

    # Draw black squares
    for r_idx in range(pattern_rows):
        for c_idx in range(pattern_cols):
            if (c_idx + r_idx) % 2 == 0: # Checkboard pattern
                board[r_idx * square_size_px : (r_idx + 1) * square_size_px,
                      c_idx * square_size_px : (c_idx + 1) * square_size_px] = 0 # Black square

    # Add instructional header
    instruction_height = 100
    instructions_panel = np.full((instruction_height, img_width), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = 0 # Black

    # Text 1: Title
    text1 = "Camera Calibration Pattern"
    (tw1, th1), _ = cv2.getTextSize(text1, font, 0.8, 2)
    cv2.putText(instructions_panel, text1, ((img_width - tw1) // 2, int(instruction_height * 0.4)), 
                font, 0.8, text_color, 2)

    # Text 2: Details
    text2 = f"{squares_x}x{squares_y} internal corners, Square Size: {CHESSBOARD_SQUARE_SIZE_MM}mm"
    (tw2, th2), _ = cv2.getTextSize(text2, font, 0.6, 2)
    cv2.putText(instructions_panel, text2, ((img_width - tw2) // 2, int(instruction_height * 0.8)), 
                font, 0.6, text_color, 2)
    
    # Combine instructions panel and chessboard
    final_image = np.vstack((instructions_panel, board))

    if save:
        filename = os.path.join(DATA_OUTPUT_DIRECTORY, PATTERNS_DIRECTORY_NAME, 
                                f"chessboard_{squares_x}x{squares_y}_corners_{CHESSBOARD_SQUARE_SIZE_MM}mm.png")
        try:
            cv2.imwrite(filename, final_image)
            msg = f"Chessboard pattern saved to: {filename}"
        except Exception as e:
            msg = f"Error saving chessboard pattern to {filename}: {e}"
        
        if parent_tk_window:
            messagebox.showinfo("Chessboard Generation", msg, parent=parent_tk_window)
        else:
            print(msg)
    
    return final_image

## END SECTION: PATTERN GENERATION TOOLS

## SECTION: TKINTER GUI COMPONENTS

class ToolTip:
    """
    A simple tooltip class for Tkinter widgets.
    Displays a small pop-up message when the mouse hovers over a widget.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        """Displays the tooltip."""
        # Calculate position of the tooltip relative to the widget
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 0
        y += self.widget.winfo_rooty() - 40 # Position above the widget

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True) # Remove window decorations
        self.tooltip.wm_geometry(f"+{x}+{y}")

        # Choose colors based on sv-ttk theme
        if THEME_AVAILABLE and sv_ttk.get_theme() == "dark":
            bg_color = "#2b2b2b"
            fg_color = "#ffffff"
        else:
            bg_color = "#ffffe0" # Light yellow
            fg_color = "#000000" # Black

        label = ttk.Label(self.tooltip, text=self.text, justify='left',
                          background=bg_color, foreground=fg_color,
                          relief='solid', borderwidth=1,
                          padding=4)
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        """Hides the tooltip."""
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = None

class CustomInputDialog(tk.Toplevel):
    """
    A customizable input dialog for Tkinter applications.
    Allows defining multiple input fields with different types and validation rules.
    """
    def __init__(self, parent, title, fields):
        """
        Initializes the dialog.
        :param parent: The parent Tkinter window.
        :param title: The title of the dialog window.
        :param fields: A list of dictionaries, each describing an input field.
                       Example: [{"label": "Field Name", "type": "int", "initial": 100, "options": [100, 200], "validation": {"min": 0}}]
        """
        super().__init__(parent)
        self.transient(parent) # Set parent
        self.grab_set() # Make modal
        self.title(title)

        self.result = None
        self.fields = fields
        self.vars = {} # Dictionary to hold Tkinter variables for each field

        body = ttk.Frame(self, padding="10 10 10 10")
        self.initial_focus = self._create_body_widgets(body)
        body.pack(padx=5, pady=5)

        self._create_buttons()

        if self.initial_focus:
            self.initial_focus.focus_set()

        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.geometry(f"+{parent.winfo_rootx()+50}+{parent.winfo_rooty()+50}") # Position relative to parent
        self.wait_window(self) # Wait until this window is destroyed

    def _create_body_widgets(self, master):
        """Creates input widgets based on the `fields` definition."""
        initial_focus = None
        for i, field in enumerate(self.fields):
            label_text = field.get("label", "")
            var_type = field.get("type", "str")
            initial_value = field.get("initial", "")
            options = field.get("options", []) # For comboboxes

            ttk.Label(master, text=label_text).grid(row=i, column=0, sticky="w", padx=5, pady=5)

            # Choose Tkinter variable type based on 'type' field
            if var_type == "int":
                self.vars[label_text] = tk.IntVar(value=initial_value)
            elif var_type == "float":
                self.vars[label_text] = tk.DoubleVar(value=initial_value)
            else: # Default to string
                self.vars[label_text] = tk.StringVar(value=initial_value)
            
            # Create Entry or Combobox
            if options:
                widget = ttk.Combobox(master, textvariable=self.vars[label_text], values=options, state="readonly")
            else:
                widget = ttk.Entry(master, textvariable=self.vars[label_text])
            
            widget.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
            if not initial_focus:
                initial_focus = widget # Set focus to the first widget

        master.columnconfigure(1, weight=1) # Make the input column expandable
        return initial_focus

    def _create_buttons(self):
        """Creates OK and Cancel buttons."""
        box = ttk.Frame(self)
        ok_button = ttk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        ok_button.pack(side=tk.LEFT, padx=5, pady=5)
        cancel_button = ttk.Button(box, text="Cancel", width=10, command=self.cancel)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def ok(self, event=None):
        """Handles the OK button press, validates input, and applies."""
        if not self.validate():
            self.initial_focus.focus_set() # Re-focus on error
            return
        self.withdraw() # Hide window
        self.update_idletasks()
        self._apply_result()
        self.cancel() # Destroy window

    def cancel(self, event=None):
        """Handles the Cancel button press or window close."""
        self.master.focus_set() # Return focus to parent
        self.destroy()

    def validate(self):
        """Validates inputs based on rules defined in `fields`."""
        for field in self.fields:
            label_text = field.get("label")
            var = self.vars[label_text]
            rules = field.get("validation", {})
            try:
                val = var.get()
                if "min" in rules and val < rules["min"]:
                    messagebox.showwarning("Validation Error", f"'{label_text}' must be greater than or equal to {rules['min']}.", parent=self)
                    return False
                if "max" in rules and val > rules["max"]:
                    messagebox.showwarning("Validation Error", f"'{label_text}' must be less than or equal to {rules['max']}.", parent=self)
                    return False
            except tk.TclError:
                messagebox.showwarning("Validation Error", f"Invalid input for '{label_text}'. Please enter a valid number.", parent=self)
                return False
        return True

    def _apply_result(self):
        """Stores the validated input values in `self.result`."""
        self.result = {label: var.get() for label, var in self.vars.items()}

class OfflineProcessingGUI(tk.Toplevel):
    """
    Tkinter Toplevel window for managing offline video processing.
    Allows selecting a video, configuring LED detection for it, and starting processing.
    """
    def __init__(self, master):
        super().__init__(master)
        self.title("Offline Video Processor")
        self.geometry("700x450")
        self.transient(master)
        self.grab_set()

        self.temp_led_roi = None # ROI specific to the selected video (if defined)
        self.temp_led_baseline = None # Baseline brightness specific to the selected video

        ttk.Label(self, text="Select a video file for offline ArUco processing.").pack(pady=(10,5))

        # File selection frame
        file_frame = ttk.Frame(self, padding="10 5")
        file_frame.pack(pady=5, padx=10, fill=tk.X)
        self.filepath_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.filepath_var, width=50, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.browse_button = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT)
        ToolTip(self.browse_button, "Browse for a video file (e.g., .mp4, .mov, .avi).")

        # LED Synchronization frame
        led_frame = ttk.LabelFrame(self, text="LED Synchronization", padding="10")
        led_frame.pack(pady=5, padx=10, fill=tk.X)
        self.led_roi_button = ttk.Button(led_frame, text="Define LED Area for this Video", command=self.define_led_roi_for_video, state=tk.DISABLED)
        self.led_roi_button.pack(pady=5, fill=tk.X)
        ToolTip(self.led_roi_button, "Opens the first frame of the selected video to define the LED's location and 'off' state.\nRequired if LED detection is enabled in Settings for this processing run.")
        self.led_status_label = ttk.Label(led_frame, text="LED area not defined for this video.")
        self.led_status_label.pack(pady=(0, 5))

        # Processing options frame
        options_frame = ttk.Frame(self, padding="10 0")
        options_frame.pack(pady=5, padx=10, fill=tk.X)
        self.ref_marker_var = tk.BooleanVar(value=USE_REFERENCE_MARKER) # Start with global setting
        self.ref_marker_check = ttk.Checkbutton(options_frame, text=f"Use Reference Marker (ID: {REFERENCE_MARKER_ID})", variable=self.ref_marker_var)
        self.ref_marker_check.pack(pady=5, anchor='w')
        ToolTip(self.ref_marker_check, "Calculate buoy poses relative to the reference marker (if detected in video).")

        # Optional: Video start timestamp
        time_input_frame = ttk.LabelFrame(self, text="Optional: Video Start Timestamp (for absolute logs)", padding="10")
        time_input_frame.pack(pady=5, padx=10, fill=tk.X)
        self.video_start_date_var = tk.StringVar()
        self.video_start_time_var = tk.StringVar()
        ttk.Label(time_input_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(time_input_frame, textvariable=self.video_start_date_var, width=15).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(time_input_frame, text="Start Time (HH:MM:SS):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(time_input_frame, textvariable=self.video_start_time_var, width=15).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        time_input_frame.columnconfigure(1, weight=1)
        ToolTip(time_input_frame, "If provided, timestamps in the CSV log file will be absolute (e.g., '2023-10-27 14:35:01.123').\nOtherwise, they will be relative to the video start (e.g., '00:00:01.123').")

        # Start Processing button
        self.process_button = ttk.Button(self, text="Start Processing", command=self.start_processing, state=tk.DISABLED, style="Accent.TButton")
        self.process_button.pack(pady=15, ipady=5)
        ToolTip(self.process_button, "Begin the offline video analysis. This may take some time.")

        # Status and progress bar
        self.status_label = ttk.Label(self, text="Status: Idle", anchor='w')
        self.status_label.pack(pady=5, fill=tk.X, padx=10)
        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(pady=(5,10), fill=tk.X, padx=10)

    def browse_file(self):
        """Opens a file dialog to select a video file."""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mov *.mp4 *.avi *.mkv"), ("All files", "*.*")),
            parent=self
        )
        if filename:
            self.filepath_var.set(filename)
            self.process_button.config(state=tk.NORMAL)
            self.led_roi_button.config(state=tk.NORMAL) # Enable LED ROI button once video is selected
            self.status_label.config(text=f"Selected: {os.path.basename(filename)}")
            self.progress_bar["value"] = 0
            
            # Reset LED ROI and baseline for new video
            self.temp_led_roi = None
            self.temp_led_baseline = None
            self.led_status_label.config(text="LED area not defined for this video.")

    def define_led_roi_for_video(self):
        """
        Allows the user to define an ROI for LED detection on the first frame of the selected video.
        Calculates a baseline brightness from this ROI when the LED is presumed OFF.
        """
        video_path = self.filepath_var.get()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file first.", parent=self)
            return

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Video Error", "Could not read the first frame of the video. Cannot define LED area.", parent=self)
            return
        
        messagebox.showinfo("Define LED Area", 
                            "The first frame of your video will be shown.\n\n"
                            "1. IMPORTANT: Ensure the LED is OFF in this frame.\n"
                            "2. Click and drag to draw a box around the LED.\n"
                            "3. Press ENTER or SPACE to confirm your selection.\n"
                            "4. Press 'c' or ESC to cancel selection.", 
                            parent=self)

        window_name = "Select LED Area (on first frame) - Press ENTER to Confirm"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Use OpenCV's selectROI to get user input
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name) # Close the ROI selection window

        if roi and roi[2] > 0 and roi[3] > 0: # Check if a valid ROI was selected
            self.temp_led_roi = roi
            x, y, w, h = [int(v) for v in roi]
            roi_patch = frame[y:y+h, x:x+w]
            self.temp_led_baseline = np.mean(roi_patch[:, :, 2]) # Mean of Red channel
            self.led_status_label.config(text=f"LED area SET for this video. Baseline (Red): {self.temp_led_baseline:.2f}")
        else:
            self.led_status_label.config(text="LED area selection cancelled.")
            self.temp_led_roi = None
            self.temp_led_baseline = None

    def start_processing(self):
        """Initiates the offline video processing."""
        video_path = self.filepath_var.get()
        if not video_path:
            messagebox.showerror("Error", "No video file selected for processing.", parent=self)
            return
        
        # Check LED settings and warn if necessary
        if ENABLE_LED_DETECTION and self.temp_led_roi is None:
            if not messagebox.askyesno("Warning", 
                                       "LED Detection is enabled in settings, but no LED area has been defined for this video.\n"
                                       "LED synchronization data will NOT be generated for this run unless you define an area.\n\n"
                                       "Do you want to continue processing WITHOUT LED detection for this video?", 
                                       icon='warning', parent=self):
                return # User cancelled

        # Get current settings from GUI
        use_ref_for_this_run = self.ref_marker_var.get()
        video_start_date = self.video_start_date_var.get()
        video_start_time = self.video_start_time_var.get()

        # Disable GUI elements during processing
        self.process_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.ref_marker_check.config(state=tk.DISABLED)
        self.led_roi_button.config(state=tk.DISABLED)
        
        # Also disable entry fields within the time input frame
        for child in self.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Entry):
                        widget.config(state=tk.DISABLED)

        self.status_label.config(text="Status: Processing video... Please wait.")
        self.progress_bar["value"] = 0
        self.update_idletasks() # Update GUI immediately

        try:
            # Call the core offline processing function
            process_video_offline(
                video_path,
                use_ref_for_this_run,
                self.status_label,
                self.progress_bar,
                video_start_date,
                video_start_time,
                self.temp_led_roi, # Pass the video-specific ROI
                self.temp_led_baseline # Pass the video-specific baseline
            )
        except Exception as e:
            messagebox.showerror("Processing Error", f"An unexpected error occurred during offline processing: {e}", parent=self)
            self.status_label.config(text=f"Status: Error - {e}")
        finally:
            # Re-enable GUI elements after processing (or error)
            self.process_button.config(state=tk.NORMAL if self.filepath_var.get() else tk.DISABLED)
            self.browse_button.config(state=tk.NORMAL)
            self.ref_marker_check.config(state=tk.NORMAL)
            self.led_roi_button.config(state=tk.NORMAL)
            for child in self.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for widget in child.winfo_children():
                        if isinstance(widget, ttk.Entry):
                            widget.config(state=tk.NORMAL)

class SettingsDialog(tk.Toplevel):
    """
    Tkinter Toplevel window for configuring application settings.
    Allows user to modify global constants like camera index, marker size, etc.
    """
    def __init__(self, master):
        super().__init__(master)
        self.title("Application Settings")
        self.transient(master)
        self.grab_set()

        # Store initial reference marker ID for validation (cannot be changed if not using ref marker)
        self.initial_ref_marker_id = REFERENCE_MARKER_ID

        # Tkinter variables to bind to widgets
        self.use_ref_marker_var = tk.BooleanVar(value=USE_REFERENCE_MARKER)
        self.ref_marker_id_var = tk.IntVar(value=REFERENCE_MARKER_ID)
        self.enable_ref_filter_var = tk.BooleanVar(value=ENABLE_REFERENCE_FILTER)
        self.ref_filter_alpha_t_var = tk.DoubleVar(value=REF_FILTER_ALPHA_T)
        self.ref_filter_alpha_r_var = tk.DoubleVar(value=REF_FILTER_ALPHA_R)
        
        self.record_video_var = tk.BooleanVar(value=RECORD_VIDEO)
        self.reverse_display_var = tk.BooleanVar(value=REVERSE_CAMERA_DISPLAY)
        self.log_all_marker_data_var = tk.BooleanVar(value=LOG_ALL_MARKER_DATA)
        self.camera_index_var = tk.IntVar(value=CAMERA_INDEX)
        self.marker_size_var = tk.DoubleVar(value=MARKER_SIZE_METERS)
        self.cam_res_width_var = tk.IntVar(value=CAMERA_RESOLUTION[0])
        self.cam_res_height_var = tk.IntVar(value=CAMERA_RESOLUTION[1])
        self.show_plots_var = tk.BooleanVar(value=SHOW_REALTIME_PLOTS)
        self.enable_led_detection_var = tk.BooleanVar(value=ENABLE_LED_DETECTION)


        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Visualization Settings
        vis_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        vis_frame.pack(fill=tk.X, pady=(0, 10))
        c1 = ttk.Checkbutton(vis_frame, variable=self.show_plots_var, text="Enable real-time plots panel")
        c1.pack(anchor='w', pady=2)
        ToolTip(c1, "Show/hide live data plots (position and orientation history) alongside the video feed.")
        c2 = ttk.Checkbutton(vis_frame, variable=self.reverse_display_var, text="Reverse live camera display horizontally")
        c2.pack(anchor='w', pady=2)
        ToolTip(c2, "Flips the live camera view horizontally for easier viewing, e.g., if your camera output is mirrored.")

        # Tracking & Logging Settings
        log_frame = ttk.LabelFrame(main_frame, text="Tracking & Logging", padding="10")
        log_frame.pack(fill=tk.X, pady=5)
        
        # Reference Marker Options
        c3 = ttk.Checkbutton(log_frame, variable=self.use_ref_marker_var, text="Enable Reference Marker by default", command=self._toggle_ref_widgets)
        c3.pack(anchor='w', pady=2)
        ToolTip(c3, "If enabled, buoy poses will be reported relative to a static world-frame marker (REFERENCE_MARKER_ID).\nThis applies to live tracking and can be overridden for offline processing.")
        
        ref_options_frame = ttk.Frame(log_frame)
        ref_options_frame.pack(fill=tk.X, padx=20) # Indent these options
        
        ttk.Label(ref_options_frame, text="Reference Marker ID:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.ref_id_entry = ttk.Entry(ref_options_frame, textvariable=self.ref_marker_id_var, width=7)
        self.ref_id_entry.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        ToolTip(self.ref_id_entry, "The ArUco ID of the static marker used as the world reference frame origin.")
        
        self.ref_filter_check = ttk.Checkbutton(ref_options_frame, variable=self.enable_ref_filter_var, text="Enable Low-Pass Filter for Reference Marker", command=self._toggle_ref_widgets)
        self.ref_filter_check.grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        ToolTip(self.ref_filter_check, "Applies a smoothing filter to the reference marker's pose estimates to reduce jitter.\nRecommended for stable world reference.")
        
        alpha_frame = ttk.Frame(ref_options_frame)
        alpha_frame.grid(row=2, column=0, columnspan=2, sticky='w', padx=20) # Indent filter alpha options
        ttk.Label(alpha_frame, text="Translation Alpha (0-1):").pack(side=tk.LEFT)
        self.ref_alpha_t_entry = ttk.Entry(alpha_frame, textvariable=self.ref_filter_alpha_t_var, width=6)
        self.ref_alpha_t_entry.pack(side=tk.LEFT, padx=(0,10))
        ToolTip(self.ref_alpha_t_entry, "Smoothing factor for reference marker translation (0 = heavily smoothed, 1 = no smoothing).")
        ttk.Label(alpha_frame, text="Rotation Alpha (0-1):").pack(side=tk.LEFT)
        self.ref_alpha_r_entry = ttk.Entry(alpha_frame, textvariable=self.ref_filter_alpha_r_var, width=6)
        self.ref_alpha_r_entry.pack(side=tk.LEFT)
        ToolTip(self.ref_alpha_r_entry, "Smoothing factor for reference marker rotation (0 = heavily smoothed, 1 = no smoothing).")

        c4 = ttk.Checkbutton(log_frame, variable=self.record_video_var, text="Record annotated video by default (Live mode)")
        c4.pack(anchor='w', pady=2)
        ToolTip(c4, "If checked, live tracking sessions will start recording an annotated video automatically.\nRecording can be toggled during live session by pressing 'V'.")
        c5 = ttk.Checkbutton(log_frame, variable=self.log_all_marker_data_var, text="Log all individual marker poses (creates extra CSV)")
        c5.pack(anchor='w', pady=2)
        ToolTip(c5, "In addition to buoy CoG data, generates a separate CSV log file for every detected individual ArUco marker's pose.")
        c6 = ttk.Checkbutton(log_frame, variable=self.enable_led_detection_var, text="Enable Red LED synchronization detection")
        c6.pack(anchor='w', pady=2)
        ToolTip(c6, "Enables monitoring a specific region of interest (ROI) for a flashing red LED to log sync events.\nThe LED ROI must be defined from the 'Tools' menu.")

        # Hardware Parameters
        cam_frame = ttk.LabelFrame(main_frame, text="Hardware Parameters", padding="10")
        cam_frame.pack(fill=tk.X, pady=5)
        
        grid_frame = ttk.Frame(cam_frame)
        grid_frame.pack(fill=tk.X)

        ttk.Label(grid_frame, text="Camera Index:").grid(row=0, column=0, sticky='w', pady=2)
        e1 = ttk.Entry(grid_frame, textvariable=self.camera_index_var, width=10)
        e1.grid(row=0, column=1, sticky='w')
        ToolTip(e1, "The numerical index of your camera device in the system (e.g., 0 for default, 1 for second).")

        ttk.Label(grid_frame, text="ArUco Marker Physical Size (meters):").grid(row=1, column=0, sticky='w', pady=2)
        e2 = ttk.Entry(grid_frame, textvariable=self.marker_size_var, width=10)
        e2.grid(row=1, column=1, sticky='w')
        ToolTip(e2, "The actual physical side length of your square ArUco markers, in meters. Crucial for accurate pose estimation.")

        ttk.Label(grid_frame, text="Desired Camera Resolution (WxH):").grid(row=2, column=0, sticky='w', pady=2)
        res_frame = ttk.Frame(grid_frame)
        res_frame.grid(row=2, column=1, sticky='w')
        e3 = ttk.Entry(res_frame, textvariable=self.cam_res_width_var, width=7)
        e3.pack(side=tk.LEFT)
        ttk.Label(res_frame, text=" x ").pack(side=tk.LEFT)
        e4 = ttk.Entry(res_frame, textvariable=self.cam_res_height_var, width=7)
        e4.pack(side=tk.LEFT)
        ToolTip(res_frame, "The preferred capture resolution for the camera. Actual resolution may vary based on camera capabilities.")
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15,5))
        ttk.Button(button_frame, text="Apply & Save Settings", command=self.apply_settings, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)

        # Initial state update for reference marker related widgets
        self._toggle_ref_widgets()

    def _toggle_ref_widgets(self):
        """Enables/disables reference marker related widgets based on master checkbox."""
        ref_enabled = self.use_ref_marker_var.get()
        filter_enabled = self.enable_ref_filter_var.get()

        self.ref_id_entry.config(state=tk.NORMAL if ref_enabled else tk.DISABLED)
        self.ref_filter_check.config(state=tk.NORMAL if ref_enabled else tk.DISABLED)
        
        alpha_state = tk.NORMAL if (ref_enabled and filter_enabled) else tk.DISABLED
        self.ref_alpha_t_entry.config(state=alpha_state)
        self.ref_alpha_r_entry.config(state=alpha_state)

    def apply_settings(self):
        """
        Applies the settings from the dialog to the global configuration constants
        and saves them to file. Performs basic validation.
        """
        global USE_REFERENCE_MARKER, REFERENCE_MARKER_ID, RECORD_VIDEO, REVERSE_CAMERA_DISPLAY
        global CAMERA_INDEX, MARKER_SIZE_METERS, CAMERA_RESOLUTION, LOG_ALL_MARKER_DATA
        global SHOW_REALTIME_PLOTS, ENABLE_LED_DETECTION, ENABLE_REFERENCE_FILTER, REF_FILTER_ALPHA_T, REF_FILTER_ALPHA_R
        
        # Validate and get reference marker ID
        v_ref_id = self.initial_ref_marker_id # Default to previous if not using or error
        if self.use_ref_marker_var.get():
            try:
                temp_id = self.ref_marker_id_var.get()
                if temp_id < 0:
                    messagebox.showwarning("Invalid Input", "Reference Marker ID must be a non-negative integer.", parent=self)
                    self.ref_marker_id_var.set(self.initial_ref_marker_id) # Revert
                    return
                v_ref_id = temp_id
            except tk.TclError:
                messagebox.showwarning("Invalid Input", "Reference Marker ID must be an integer.", parent=self)
                self.ref_marker_id_var.set(self.initial_ref_marker_id) # Revert
                return
        
        # Validate other numerical inputs
        try:
            new_cam_idx = self.camera_index_var.get()
            new_marker_size = self.marker_size_var.get()
            new_cam_width = self.cam_res_width_var.get()
            new_cam_height = self.cam_res_height_var.get()
            new_alpha_t = self.ref_filter_alpha_t_var.get()
            new_alpha_r = self.ref_filter_alpha_r_var.get()

            if new_cam_idx < 0:
                raise ValueError("Camera index must be 0 or greater.")
            if new_marker_size <= 0:
                raise ValueError("Marker size must be greater than 0.")
            if new_cam_width <= 0 or new_cam_height <= 0:
                raise ValueError("Camera resolution dimensions must be positive integers.")
            if not (0.0 <= new_alpha_t <= 1.0):
                raise ValueError("Translation Alpha must be between 0.0 and 1.0.")
            if not (0.0 <= new_alpha_r <= 1.0):
                raise ValueError("Rotation Alpha must be between 0.0 and 1.0.")

        except (tk.TclError, ValueError) as e:
            messagebox.showwarning("Invalid Input", f"Input error: {e}", parent=self)
            return

        # Update global constants
        SHOW_REALTIME_PLOTS = self.show_plots_var.get()
        USE_REFERENCE_MARKER = self.use_ref_marker_var.get()
        REFERENCE_MARKER_ID = v_ref_id
        ENABLE_REFERENCE_FILTER = self.enable_ref_filter_var.get()
        REF_FILTER_ALPHA_T = new_alpha_t
        REF_FILTER_ALPHA_R = new_alpha_r
        RECORD_VIDEO = self.record_video_var.get()
        REVERSE_CAMERA_DISPLAY = self.reverse_display_var.get()
        LOG_ALL_MARKER_DATA = self.log_all_marker_data_var.get()
        CAMERA_INDEX = new_cam_idx
        MARKER_SIZE_METERS = new_marker_size
        CAMERA_RESOLUTION = (new_cam_width, new_cam_height)
        ENABLE_LED_DETECTION = self.enable_led_detection_var.get()

        print(f"Settings updated: SHOW_REALTIME_PLOTS={SHOW_REALTIME_PLOTS}, USE_REFERENCE_MARKER={USE_REFERENCE_MARKER}, REFERENCE_MARKER_ID={REFERENCE_MARKER_ID}, ENABLE_REFERENCE_FILTER={ENABLE_REFERENCE_FILTER}, REF_FILTER_ALPHA_T={REF_FILTER_ALPHA_T}, REF_FILTER_ALPHA_R={REF_FILTER_ALPHA_R}, RECORD_VIDEO={RECORD_VIDEO}, REVERSE_CAMERA_DISPLAY={REVERSE_CAMERA_DISPLAY}, LOG_ALL_MARKER_DATA={LOG_ALL_MARKER_DATA}, CAMERA_INDEX={CAMERA_INDEX}, MARKER_SIZE_METERS={MARKER_SIZE_METERS}, CAMERA_RESOLUTION={CAMERA_RESOLUTION}, ENABLE_LED_DETECTION={ENABLE_LED_DETECTION}")

        save_settings() # Save updated settings to file
        initialize_buoy_configs() # Re-initialize buoy config with potentially new marker sizes/ref markers
        
        messagebox.showinfo("Settings Applied", "Application settings updated and saved successfully.", parent=self)
        self.destroy() # Close the settings dialog

class BuoyTrackerApp:
    """
    Main application class for the ArUco 6-DOF Buoy Tracker.
    Manages the main Tkinter window and navigates to different functionalities.
    """
    def __init__(self, master):
        self.master = master
        master.title("ArUco-6DOF Buoy Tracker")
        master.geometry("500x550")
        master.resizable(False, False) # Fixed window size

        # Apply sv-ttk theme if available
        if THEME_AVAILABLE:
            sv_ttk.set_theme("light") # Default to light theme

        # Set application icon
        try:
            # Requires Pillow: pip install Pillow
            master.iconphoto(True, tk.PhotoImage(file=APP_ICON_PATH))
        except tk.TclError:
            print(f"Warning: Application icon '{APP_ICON_PATH}' not found or invalid. Using default icon.")
        except Exception as e:
            print(f"Warning: Could not set application icon: {e}")

        # Configure main window grid for content expansion
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        self._create_menubar()

        main_frame = ttk.Frame(master, padding="15")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1) # Center content

        # Title
        ttk.Label(main_frame, text="Buoy Tracking & Analysis Tool", font=("", 16, "bold")).pack(pady=(0, 20))

        # Core Functions
        core_frame = ttk.LabelFrame(main_frame, text="Core Functions", padding="10")
        core_frame.pack(fill=tk.X, pady=10)
        core_frame.columnconfigure(0, weight=1)

        live_btn = ttk.Button(core_frame, text="Start Live Tracking Session", command=self.ui_start_live_tracking, style="Accent.TButton")
        live_btn.pack(fill=tk.X, pady=5, ipady=8)
        ToolTip(live_btn, "Begin real-time buoy tracking using a connected camera. Data and optional video will be saved.")

        offline_btn = ttk.Button(core_frame, text="Process Recorded Video (Offline)", command=self.ui_launch_offline_processor)
        offline_btn.pack(fill=tk.X, pady=5, ipady=4)
        ToolTip(offline_btn, "Analyze a pre-recorded video file to extract buoy tracking data and generate an annotated video.")

        # Calibration & Setup Tools
        calib_frame = ttk.LabelFrame(main_frame, text="Calibration & Setup Tools", padding="10")
        calib_frame.pack(fill=tk.X, pady=10)
        calib_frame.columnconfigure(0, weight=1)
        calib_frame.columnconfigure(1, weight=1) # Two columns for buttons

        cam_calib_btn = ttk.Button(calib_frame, text="Calibrate Camera", command=self.ui_calibrate_camera)
        cam_calib_btn.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ToolTip(cam_calib_btn, "Perform camera intrinsic calibration using a chessboard pattern. Essential for accurate pose estimation.")

        buoy_calib_btn = ttk.Button(calib_frame, text="Calibrate Buoy Geometry", command=self.ui_calibrate_buoy_geometry)
        buoy_calib_btn.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ToolTip(buoy_calib_btn, "Calibrate the precise relative positions of markers on a specific buoy. Improves buoy CoG accuracy.")
        
        # LED ROI definition is duplicated here for quick access
        led_roi_btn = ttk.Button(calib_frame, text="Define LED Area (Live Sync)", command=self.ui_define_led_roi_live)
        led_roi_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(10,5))
        ToolTip(led_roi_btn, "Define the screen area to monitor for a red LED to synchronize data logging.\nDo this while the LED is OFF to set a proper baseline.")


        # Status Bar
        self.status_var = tk.StringVar(value="Ready. Welcome!")
        status_bar = ttk.Frame(master, style="Card.TFrame", padding=5) # Use a Card style for status bar
        status_bar.grid(row=1, column=0, sticky="ew")
        ttk.Label(status_bar, textvariable=self.status_var).pack(anchor='w')

        # Ensure output directories exist at startup
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """Creates necessary output directories if they don't exist."""
        os.makedirs(DATA_OUTPUT_DIRECTORY, exist_ok=True)
        os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME), exist_ok=True)
        os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME), exist_ok=True)
        os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, PATTERNS_DIRECTORY_NAME), exist_ok=True)
        print("Required directories ensured to exist.")

    def _create_menubar(self):
        """Creates the application's menu bar."""
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings...", command=self.ui_configure_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)

        # Tools Menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # Generate Pattern Sub-menu
        gen_menu = tk.Menu(tools_menu, tearoff=0)
        tools_menu.add_cascade(label="Generate Pattern", menu=gen_menu)
        gen_menu.add_command(label="Single ArUco Marker...", command=self.ui_generate_single_marker)
        gen_menu.add_command(label="Reference Marker (Global ID)...", command=self.ui_generate_global_reference_marker)
        gen_menu.add_command(label="Chessboard Calibration Pattern...", command=self.ui_generate_chessboard_pattern)
        
        tools_menu.add_separator()
        tools_menu.add_command(label="Define LED Area (Live Sync)...", command=self.ui_define_led_roi_live)

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        if THEME_AVAILABLE:
            help_menu.add_command(label="Toggle Dark/Light Mode", command=sv_ttk.toggle_theme)
        help_menu.add_command(label="About", command=self.ui_show_about)

    def ui_show_about(self):
        """Displays the 'About' dialog."""
        messagebox.showinfo("About ArUco-6DOF Buoy Tracker",
                            f"ArUco-Based 6-DOF Buoy Tracker\n"
                            f"Version: 1.3\n"
                            f"Author: Paul FROMONT\n"
                            f"Contact: paul.fromont@polytechnique.org\n\n"
                            f"This application provides real-time 6-DOF (position and orientation) tracking of buoys "
                            f"equipped with ArUco markers, using a standard webcam. It supports data logging, "
                            f"offline video processing, and calibration tools.",
                            parent=self.master)

    def ui_generate_single_marker(self):
        """Opens a dialog to generate a single ArUco marker."""
        fields = [
            {"label": "Marker ID", "type": "int", "initial": 0, "validation": {"min": 0, "max": 249}},
            {"label": "Image Size (pixels)", "type": "int", "initial": 600, "validation": {"min": 100}},
            {"label": "Border Width (pixels)", "type": "int", "initial": 50, "validation": {"min": 10}}
        ]
        dialog = CustomInputDialog(self.master, "Generate Single ArUco Marker", fields)
        if dialog.result:
            marker_id, img_size_px, border_px = dialog.result.values()
            self.status_var.set(f"Generating marker ID {marker_id}...")
            self.master.update_idletasks() # Update GUI
            
            generated_img = generate_aruco_marker(marker_id, img_size_px, border_px, True, self.master)
            if generated_img is not None:
                # Show generated marker temporarily in an OpenCV window
                cv2.imshow(f"Generated ArUco Marker - ID:{marker_id}", generated_img)
                cv2.waitKey(0) # Wait for any key press
                cv2.destroyWindow(f"Generated ArUco Marker - ID:{marker_id}")
            self.status_var.set("Ready.")

    def ui_generate_global_reference_marker(self):
        """Opens a dialog to generate the globally configured reference ArUco marker."""
        if not USE_REFERENCE_MARKER:
            if not messagebox.askyesno("Reference Marker Disabled", 
                                       f"The global setting 'Use Reference Marker' is currently disabled. "
                                       f"The generated marker (ID {REFERENCE_MARKER_ID}) will still be created, "
                                       f"but it won't be used as a reference unless you enable the setting. "
                                       f"Do you wish to proceed?", icon='info', parent=self.master):
                return
        
        fields = [
            {"label": "Image Size (pixels)", "type": "int", "initial": 600, "validation": {"min": 100}},
            {"label": "Border Width (pixels)", "type": "int", "initial": 50, "validation": {"min": 10}}
        ]
        dialog = CustomInputDialog(self.master, f"Generate Global Reference Marker (ID: {REFERENCE_MARKER_ID})", fields)
        if dialog.result:
            img_size_px, border_px = dialog.result.values()
            self.status_var.set(f"Generating global reference marker ID {REFERENCE_MARKER_ID}...")
            self.master.update_idletasks()
            
            generated_img = generate_aruco_marker(REFERENCE_MARKER_ID, img_size_px, border_px, True, self.master)
            if generated_img is not None:
                cv2.imshow(f"Generated Reference Marker - ID:{REFERENCE_MARKER_ID}", generated_img)
                cv2.waitKey(0)
                cv2.destroyWindow(f"Generated Reference Marker - ID:{REFERENCE_MARKER_ID}")
            self.status_var.set("Ready.")

    def ui_generate_chessboard_pattern(self):
        """Opens a dialog to generate a chessboard calibration pattern."""
        # Chessboard dimensions are fixed by CALIBRATION_CHESSBOARD_SHAPE
        squares_x, squares_y = CALIBRATION_CHESSBOARD_SHAPE
        fields = [
            {"label": f"Square Size (pixels) ({squares_x}x{squares_y} internal corners)", "type": "int", "initial": 100, "validation": {"min": 50}}
        ]
        dialog = CustomInputDialog(self.master, f"Generate Chessboard Pattern", fields)
        if dialog.result:
            square_size_px = list(dialog.result.values())[0]
            self.status_var.set("Generating chessboard pattern...")
            self.master.update_idletasks()
            
            generated_img = create_chessboard(squares_x, squares_y, square_size_px, True, self.master)
            if generated_img is not None:
                cv2.imshow(f"Generated Chessboard Pattern ({squares_x}x{squares_y} internal corners)", generated_img)
                cv2.waitKey(0)
                cv2.destroyWindow(f"Generated Chessboard Pattern ({squares_x}x{squares_y} internal corners)")
            self.status_var.set("Ready.")

    def ui_calibrate_camera(self):
        """Initiates the camera calibration process."""
        self.status_var.set("Starting camera calibration process...")
        self.master.update_idletasks()
        
        messagebox.showinfo("Camera Calibration Instructions", 
                            "A camera view will open. Follow the instructions within the camera window:\n\n"
                            "- Aim your camera at a printed chessboard pattern.\n"
                            "- Move the chessboard to capture various angles and distances.\n"
                            "- Press 'c' to capture a good, clear view of the chessboard.\n"
                            "- Press 'q' to finish capturing images and start the calibration calculation.", 
                            parent=self.master)
        
        calibrate_camera(self.master) # Call the core calibration function
        self.status_var.set("Ready.")

    def ui_define_led_roi_live(self):
        """
        Guides the user through defining an LED detection ROI for live tracking.
        This sets the global LED_DETECTION_ROI and LED_OFF_BASELINE_BRIGHTNESS.
        """
        global LED_DETECTION_ROI, LED_OFF_BASELINE_BRIGHTNESS
        
        if not ENABLE_LED_DETECTION:
            messagebox.showwarning("Feature Disabled", 
                                   "LED detection is currently disabled in the application settings. "
                                   "Please enable it under 'File -> Settings' before defining the LED area.", 
                                   parent=self.master)
            return

        messagebox.showinfo("Define LED Area (Live Synchronization)", 
                            "A live camera view will open.\n\n"
                            "1. Position your camera so that the RED LED (if applicable) is clearly visible and OFF.\n"
                            "2. Press 's' on your keyboard to freeze the current frame.\n"
                            "3. In the frozen frame, click and drag a box precisely around the LED.\n"
                            "4. Press ENTER or SPACE to confirm your selection, or ESC to cancel.", 
                            parent=self.master)
        
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera {CAMERA_INDEX}. Check connection.", parent=self.master)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

        window_name = "Live ROI Selection - Press 's' to freeze and select"
        frozen_frame = None
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Camera Error", "Failed to grab a frame from camera during LED ROI setup.", parent=self.master)
                break
            
            display_frame = frame.copy()
            h, w, _ = display_frame.shape
            
            cv2.putText(display_frame, "Press 's' to freeze frame for ROI selection (LED should be OFF)", (20, h - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE * 0.7, (0, 255, 255), OVERLAY_THICKNESS)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                frozen_frame = frame.copy() # Capture the frame
                break
            if key in [ord('c'), 27]: # 'c' or ESC to cancel
                break
        
        cap.release() # Release camera after freezing frame or cancelling
        cv2.destroyAllWindows() # Close the live preview window

        if frozen_frame is not None:
            # If a frame was successfully frozen, proceed to ROI selection
            roi_window_name = "Draw Box around LED (OFF STATE), Press ENTER to Confirm"
            cv2.namedWindow(roi_window_name, cv2.WINDOW_NORMAL)
            roi = cv2.selectROI(roi_window_name, frozen_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(roi_window_name) # Close the ROI selection window

            if roi and roi[2] > 0 and roi[3] > 0:
                LED_DETECTION_ROI = roi # Store the ROI globally
                x, y, w, h = [int(v) for v in roi]
                roi_patch = frozen_frame[y:y+h, x:x+w]
                LED_OFF_BASELINE_BRIGHTNESS = np.mean(roi_patch[:, :, 2]) # Calculate baseline from red channel
                
                save_settings() # Save the updated global settings
                
                self.status_var.set(f"Global LED ROI set. Baseline brightness: {LED_OFF_BASELINE_BRIGHTNESS:.2f}")
                messagebox.showinfo("LED Area Defined", 
                                    f"Global LED detection area defined and saved.\n"
                                    f"The 'OFF' baseline brightness for the red channel is: {LED_OFF_BASELINE_BRIGHTNESS:.2f}.\n"
                                    f"This will be used for LED synchronization in live and offline modes.", 
                                    parent=self.master)
            else:
                self.status_var.set("Global LED ROI selection cancelled.")
        else:
            self.status_var.set("Global LED ROI selection cancelled (no frame captured).")
        
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()

    def ui_start_live_tracking(self):
        """Initiates a live buoy tracking session."""
        self.status_var.set("Preparing live tracking session...")
        self.master.update_idletasks() # Update GUI to show status

        # Confirm settings before starting
        log_all_text = "Enabled" if LOG_ALL_MARKER_DATA else "Disabled"
        led_sync_text = "Enabled"
        if not ENABLE_LED_DETECTION:
            led_sync_text = "Disabled"
        elif ENABLE_LED_DETECTION and LED_DETECTION_ROI is None:
            led_sync_text += " (WARNING: Area not defined, sync will not work!)"

        conf_msg = (f"Start live tracking session with the following settings?\n\n"
                    f"  • Camera Index: {CAMERA_INDEX} @ {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]} px\n"
                    f"  • Reference Frame: {'Enabled (ID ' + str(REFERENCE_MARKER_ID) + ')' if USE_REFERENCE_MARKER else 'Disabled'}\n"
                    f"  • LED Synchronization: {led_sync_text}\n"
                    f"  • Auto-record Video: {'Enabled' if RECORD_VIDEO else 'Disabled'}\n"
                    f"  • Log All Markers: {log_all_text}\n\n"
                    f"A new OpenCV window will open for tracking. Press 'Q' to quit.")
        
        if messagebox.askyesno("Confirm Live Tracking Session", conf_msg, parent=self.master):
            self.status_var.set("Live tracking active... (See OpenCV window)")
            self.master.update_idletasks()
            try:
                track_buoy_with_aruco_6dof(self.master) # Call the core live tracking function
            except Exception as e:
                messagebox.showerror("Live Tracking Error", f"An unexpected error occurred during live tracking: {e}", parent=self.master)
                self.status_var.set(f"Live tracking error: {e}")
            finally:
                self.status_var.set("Ready.") # Reset status after session ends
        else:
            self.status_var.set("Live tracking session cancelled.")

    def ui_launch_offline_processor(self):
        """Launches the dedicated GUI for offline video processing."""
        self.status_var.set("Opening Offline Video Processor...")
        self.master.update_idletasks()
        
        # Create a new Toplevel window for the offline processor
        offline_processor_gui = OfflineProcessingGUI(self.master)
        # Handle window close: reset status label
        offline_processor_gui.protocol("WM_DELETE_WINDOW", lambda: [self.status_var.set("Ready."), offline_processor_gui.destroy()])

    def ui_configure_settings(self):
        """Launches the settings configuration dialog."""
        self.status_var.set("Opening application settings...")
        self.master.update_idletasks()
        
        settings_dialog = SettingsDialog(self.master)
        self.master.wait_window(settings_dialog) # Wait until settings dialog is closed
        
        self.status_var.set("Settings window closed. Changes applied.")

    def ui_calibrate_buoy_geometry(self):
        """Initiates the buoy marker geometry calibration process."""
        buoy_ids_configured = list(BUOY_ID_TO_MARKER_IDS_MAP.keys())
        if not buoy_ids_configured:
            messagebox.showerror("Configuration Error", "No buoys are defined in the application configuration (BUOY_ID_TO_MARKER_IDS_MAP). Cannot proceed with buoy geometry calibration.", parent=self.master)
            return

        # Suggest a number of frames to capture based on the minimum required samples per pair
        # For a 3-marker buoy, we need to calibrate X1 and X2 relative to X0.
        # If BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR is 5, and there are 2 other markers,
        # we need 5 good captures for X1 and 5 for X2. So at least 5 frames where both X0,X1,X2 are visible.
        # Suggesting more than minimal for better accuracy.
        num_other_markers = len(BUOY_ID_TO_MARKER_IDS_MAP.get(buoy_ids_configured[0], [])) - 1
        num_frames_suggestion = BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR * max(1, num_other_markers) * 2 # Aim for double the minimum for robustness

        fields = [
            {"label": "Buoy ID to Calibrate", "type": "int", "initial": buoy_ids_configured[0], "options": buoy_ids_configured},
            {"label": "Total Views to Capture", "type": "int", "initial": max(10, num_frames_suggestion), "validation": {"min": 5}}
        ]
        dialog = CustomInputDialog(self.master, "Calibrate Buoy Marker Geometry", fields)
        if dialog.result:
            buoy_id_to_calibrate, num_frames_to_capture = dialog.result.values()
            
            self.status_var.set(f"Starting geometry calibration for Buoy ID: {buoy_id_to_calibrate}...")
            self.master.update_idletasks()
            
            calibrate_buoy_marker_geometry(buoy_id_to_calibrate, self.master, num_frames_to_capture=num_frames_to_capture)
            self.status_var.set(f"Buoy {buoy_id_to_calibrate} geometry calibration finished. Status: Ready.")

## END SECTION: TKINTER GUI COMPONENTS

if __name__ == "__main__":
    # Ensure base directories exist before anything else
    os.makedirs(DATA_OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME), exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, CONFIG_DIRECTORY_NAME), exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, PATTERNS_DIRECTORY_NAME), exist_ok=True)
    
    # Load settings at application startup
    load_settings()
    
    # Re-initialize buoy configurations based on potentially new settings/calibrations
    # This is important to ensure MARKER_BUOY_GEOMETRY_CONFIG is up-to-date
    initialize_buoy_configs() 

    print("Application starting...")
    root = tk.Tk()
    app = BuoyTrackerApp(root)
    root.mainloop()
    print("Application closed.")
