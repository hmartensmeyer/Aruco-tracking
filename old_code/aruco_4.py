import cv2
import numpy as np
from cv2 import aruco
import time
import csv
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from scipy.spatial.transform import Rotation as R # Crucial for quaternion and rotation math
import json # For storing calibration data
import shutil # For copying files

#################################
# CONFIGURATION PARAMETERS
#################################

# Camera Setup
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (3840, 2160) 
CAMERA_FPS = 30
REVERSE_CAMERA_DISPLAY = True

# Video Recording Settings
RECORD_VIDEO = False 
VIDEO_CODEC = 'mp4v'
VIDEO_FPS = 30
VIDEO_EXTENSION = '.mp4'

# ArUco Marker Configuration
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
MARKER_SIZE_METERS = 0.1577

# Reference Marker Configuration
REFERENCE_MARKER_ID = 0
USE_REFERENCE_MARKER = True

# Buoy Configuration
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
USE_TRIANGULAR_MARKER_FUSION = True 
TRIANGLE_SIDE_LENGTH_METERS = 0.21 

BUOY_COG_OFFSET_METERS = 0.23168 
BUOY_COG_OFFSET_VECTOR_ASSEMBLY_FRAME = np.array([0, -BUOY_COG_OFFSET_METERS, -1/3*TRIANGLE_SIDE_LENGTH_METERS], dtype=np.float32)

_s = TRIANGLE_SIDE_LENGTH_METERS
MARKER_RELATIVE_GEOMETRY = { 
    0: { 'name': 'X0', 'rel_tvec_to_X0': np.array([0.0, 0.0, 0.0], dtype=np.float32), 'rel_rvec_to_X0': np.array([0.0, 0.0, 0.0], dtype=np.float32)},
    1: { 'name': 'X1', 'rel_tvec_to_X0': np.array([_s, 0.0, 0.0], dtype=np.float32), 'rel_rvec_to_X0': np.array([0.0, np.deg2rad(120), 0.0], dtype=np.float32)},
    2: { 'name': 'X2', 'rel_tvec_to_X0': np.array([_s * np.cos(np.pi/3), 0.0, _s * np.sin(np.pi/3)], dtype=np.float32), 'rel_rvec_to_X0': np.array([0.0, np.deg2rad(240), 0.0], dtype=np.float32)}
}

DEFAULT_CAMERA_MATRIX = np.array([[1000, 0, CAMERA_RESOLUTION[0]/2], [0, 1000, CAMERA_RESOLUTION[1]/2], [0, 0, 1]], dtype=np.float32)
DEFAULT_DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

CALIBRATION_CHESSBOARD_SHAPE = (9, 6)
CHESSBOARD_SQUARE_SIZE_MM = 26.9
BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR = 5

TRAJECTORY_LENGTH = 30
TRAJECTORY_COLOR = (0, 0, 255)
MARKER_INFO_COLOR = (0, 255, 0)
BUOY_COG_INFO_COLOR = (255, 100, 0)
TIMESTAMP_COLOR = (0, 0, 255)
REFERENCE_MARKER_COLOR = (255, 0, 255)
RECORD_INDICATOR_COLOR = (0, 0, 255)
OFFLINE_PROGRESS_COLOR = (255, 165, 0)
SHOW_OFFLINE_PROCESSING_PREVIEW = True
OFFLINE_PREVIEW_RESIZE_FACTOR = 1

DATA_OUTPUT_DIRECTORY = "buoy_tracking_data"
CALIBRATION_FILE = "camera_calibration.npz"
EXPERIMENTS_BASE_DIRECTORY_NAME = "experiments"
BUOY_GEOMETRY_CALIBRATION_FILE = "buoy_marker_calibrations.json"

LOG_ALL_MARKER_DATA = False

# Kalman Filter Settings for Buoy CoG (ESKF using Quaternions)
USE_KALMAN_FILTER = False       
KF_PROCESS_NOISE_VEL_POS = 0.1   # Std dev of process noise for position (velocity component, m/s)
KF_PROCESS_NOISE_VEL_ORI = 0.2   # Std dev of process noise for orientation (angular velocity component, rad/s for error rotvec)
KF_MEASUREMENT_NOISE_POS = 0.02  # Std dev of measurement noise for position (m)
KF_MEASUREMENT_NOISE_ORI = 0.05  # Std dev of measurement noise for orientation (rad for error rotvec)

#################################
# KALMAN FILTER CLASS (ESKF with Quaternions)
#################################
class KalmanFilterPose6DOF_Quat:
    def __init__(self, process_noise_vel_pos, process_noise_vel_ori,
                 measurement_noise_pos, measurement_noise_ori):
        """
        Initializes a 6DOF Pose Kalman Filter using an Error State formulation
        with Quaternions for orientation.
        Nominal state: [x, y, z, qx, qy, qz, qw] (7D)
        Error state: [dp_x, dp_y, dp_z, dtheta_x, dtheta_y, dtheta_z] (6D) - error rotvec for orientation
        """
        # Nominal state: [pos (3D), quat_xyzw (4D)]
        self.x_nominal = np.array([0., 0., 0., 0., 0., 0., 1.]) 
        
        # Error state covariance (6x6 for [dp_x, dp_y, dp_z, dtheta_x, dtheta_y, dtheta_z])
        self.P_error = np.eye(6) * 1000.0  # Large initial uncertainty

        # Error state transition matrix (Identity for random walk of error)
        self.F_error = np.eye(6)
        # Error measurement matrix (Identity, as we derive error directly)
        self.H_error = np.eye(6)

        self._noise_vel_pos_sigma = process_noise_vel_pos
        self._noise_vel_ori_sigma = process_noise_vel_ori
        self.Q_error = np.zeros((6, 6)) # Process noise covariance for error state

        # Measurement noise covariance for error state
        # Assumes measurement_noise_ori is std dev for the 3D rotation vector components of error
        self.R_error = np.diag(
            [measurement_noise_pos**2] * 3 + \
            [measurement_noise_ori**2] * 3 
        )
        
        self.initialized = False

    def predict(self, dt):
        """Predicts the error state covariance. Nominal state is predicted if a motion model is used (not implemented here for simplicity)."""
        if not self.initialized:
            return

        # 1. Predict nominal state (if a dynamic model beyond random walk was used)
        # For a simple random walk of pose, nominal state doesn't change here,
        # it only gets corrected in update. If velocity was part of nominal state, it would be updated here.
        # self.x_nominal[:3] += self.x_nominal_velocity[:3] * dt 
        # self.x_nominal[3:7] = (R.from_quat(self.x_nominal[3:7]) * R.from_rotvec(self.x_nominal_ang_velocity * dt)).as_quat()

        # 2. Predict error state covariance
        # Q_error_ii = (sigma_velocity_error * dt)^2 for position error
        # Q_error_ii = (sigma_angular_velocity_error * dt)^2 for orientation error
        q_pos_val = (self._noise_vel_pos_sigma * dt)**2
        q_ori_val = (self._noise_vel_ori_sigma * dt)**2 # This is for the error rotvec
        self.Q_error = np.diag([q_pos_val]*3 + [q_ori_val]*3)
        
        self.P_error = self.F_error @ self.P_error @ self.F_error.T + self.Q_error

    def update(self, z_measurement_pose): 
        """
        Updates the nominal state and error covariance using a new pose measurement.
        Args:
            z_measurement_pose (np.array): 7D array [tx, ty, tz, qx, qy, qz, qw]
        """
        t_meas = z_measurement_pose[:3]
        q_meas = z_measurement_pose[3:7] # Expected: [qx, qy, qz, qw]

        if not self.initialized:
            self.x_nominal[:3] = t_meas.copy()
            self.x_nominal[3:7] = q_meas.copy()
            
            # Initialize P_error to be smaller, based on measurement noise
            initial_p_error_diag = np.diag(self.R_error).copy() * 0.1 
            np.fill_diagonal(self.P_error, initial_p_error_diag)
            
            self.initialized = True
            return self.get_state()

        # 1. Calculate innovation (measurement residual for the error state)
        # Position error
        delta_p_obs = t_meas - self.x_nominal[:3]

        # Orientation error:
        # q_error_obs = q_measured * q_nominal_predicted.conjugate()
        # This quaternion (q_error_obs) represents the rotation from the predicted nominal frame to the measured frame.
        
        # Ensure current nominal quaternion is valid for R object
        q_nominal_current_norm = np.linalg.norm(self.x_nominal[3:7])
        if q_nominal_current_norm < 1e-9: # Should not happen if state is managed
            q_nominal_current_scipy_obj = R.identity()
        else:
            q_nominal_current_scipy_obj = R.from_quat(self.x_nominal[3:7] / q_nominal_current_norm)
        
        q_meas_scipy_obj = R.from_quat(q_meas) # q_meas is [x,y,z,w]
        
        # Error quaternion: rotation from current nominal to measured observation
        q_error_obs_obj = q_meas_scipy_obj * q_nominal_current_scipy_obj.inv()
        delta_theta_obs = q_error_obs_obj.as_rotvec() # 3D rotation vector representing orientation error

        # Innovation vector y_error (6D)
        y_error = np.concatenate((delta_p_obs, delta_theta_obs))
        
        # 2. Standard EKF update steps for the error state
        S_error = self.H_error @ self.P_error @ self.H_error.T + self.R_error
        K_error = self.P_error @ self.H_error.T @ np.linalg.inv(S_error)
        
        # Estimated 6D error state vector
        delta_x_error_hat = K_error @ y_error
        
        # Update error covariance
        self.P_error = (np.eye(6) - K_error @ self.H_error) @ self.P_error

        # 3. Inject estimated error back into the nominal state
        # Position update
        self.x_nominal[:3] += delta_x_error_hat[:3]
        
        # Orientation update: q_new = delta_q_correction * q_old_nominal
        delta_theta_error_for_correction = delta_x_error_hat[3:6]
        delta_q_correction_obj = R.from_rotvec(delta_theta_error_for_correction)
        
        # current_q_nominal_obj was already defined above from self.x_nominal[3:7]
        updated_q_nominal_obj = delta_q_correction_obj * q_nominal_current_scipy_obj
        self.x_nominal[3:7] = updated_q_nominal_obj.as_quat()

        # Normalize the quaternion part of the nominal state
        norm_q_updated = np.linalg.norm(self.x_nominal[3:7])
        if norm_q_updated > 1e-9:
            self.x_nominal[3:7] /= norm_q_updated
        else: # Safety reset, though unlikely if inputs are sane
            self.x_nominal[3:7] = np.array([0.,0.,0.,1.])

        # The error state is conceptually reset to zero after injection for the next iteration.
        return self.get_state()

    def get_state(self):
        """Returns the current nominal state estimate: [pos (3D), quat_xyzw (4D)]."""
        return self.x_nominal.copy()

#################################
# UTILITY FUNCTIONS
#################################
def load_camera_calibration():
    # ... (no changes)
    calibration_path = os.path.join(DATA_OUTPUT_DIRECTORY, CALIBRATION_FILE)
    try:
        data = np.load(calibration_path)
        return data['camera_matrix'], data['dist_coeffs']
    except Exception:
        print(f"Calibration file not found or error loading. Using defaults: {calibration_path}")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

def get_video_writer(frame_size, full_video_path, fps=None):
    # ... (no changes)
    codec_str_global = VIDEO_CODEC
    if not (isinstance(codec_str_global, str) and len(codec_str_global) == 4):
        codec_to_try = 'mp4v'
    else:
        codec_to_try = codec_str_global

    output_fps = fps if fps is not None and fps > 0 else VIDEO_FPS
    writer = None
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_to_try)
        writer = cv2.VideoWriter(full_video_path, fourcc, output_fps, frame_size)
        if writer.isOpened():
            print(f"Video writer created for: {full_video_path} with codec {codec_to_try}.")
            return writer
        else:
            print(f"Warning: Codec {codec_to_try} failed to open writer for {full_video_path}.")
    except Exception as e:
        print(f"Error initializing writer with codec {codec_to_try} for {full_video_path}: {e}")

    print(f"Trying fallback codec XVID for {full_video_path}.")
    try:
        fourcc_xvid = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(full_video_path, fourcc_xvid, output_fps, frame_size)
        if writer.isOpened():
            print(f"Video writer created for: {full_video_path} with fallback codec XVID.")
            return writer
        else:
            print(f"Error: Failed to create video writer with primary ('{codec_to_try}') and fallback ('XVID') codecs for: {full_video_path}")
            return None
    except Exception as e:
        print(f"Error initializing writer with fallback codec XVID for {full_video_path}: {e}")
        return None

def load_buoy_geometry_calibrations():
    # ... (no changes)
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, BUOY_GEOMETRY_CALIBRATION_FILE)
    if os.path.exists(filepath):
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
            print(f"Error loading buoy geometry calibrations from {filepath}: {e}. Using defaults.")
            return {}
    print(f"Buoy geometry calibration file not found: {filepath}. Returning empty config.")
    return {}

def save_buoy_geometry_calibrations(calibrations):
    # ... (no changes)
    filepath = os.path.join(DATA_OUTPUT_DIRECTORY, BUOY_GEOMETRY_CALIBRATION_FILE)
    try:
        serializable_calibrations = {}
        for buoy_id, markers_data in calibrations.items():
            serializable_calibrations[str(buoy_id)] = {}
            for marker_id, pose_data in markers_data.items():
                serializable_marker_data = {}
                if 't_rel_to_X0' in pose_data and pose_data['t_rel_to_X0'] is not None:
                    serializable_marker_data['t_rel_to_X0'] = pose_data['t_rel_to_X0'].tolist()
                else: 
                    serializable_marker_data['t_rel_to_X0'] = None
                if 'rvec_rel_to_X0' in pose_data and pose_data['rvec_rel_to_X0'] is not None:
                    serializable_marker_data['rvec_rel_to_X0'] = pose_data['rvec_rel_to_X0'].tolist()
                else: 
                    serializable_marker_data['rvec_rel_to_X0'] = None
                serializable_calibrations[str(buoy_id)][str(marker_id)] = serializable_marker_data
        
        with open(filepath, 'w') as f:
            json.dump(serializable_calibrations, f, indent=4)
        print(f"Saved buoy geometry calibrations to: {filepath}")
    except Exception as e:
        print(f"Error saving buoy geometry calibrations to {filepath}: {e}")

ALL_BUOY_GEOMETRY_CALIBRATIONS = load_buoy_geometry_calibrations()
MARKER_BUOY_GEOMETRY_CONFIG = {}
BUOY_COG_OFFSETS_IN_BRF_CONFIG = {}

def initialize_buoy_configs():
    # ... (no changes)
    global MARKER_BUOY_GEOMETRY_CONFIG, BUOY_COG_OFFSETS_IN_BRF_CONFIG, ALL_BUOY_GEOMETRY_CALIBRATIONS
    ALL_BUOY_GEOMETRY_CALIBRATIONS = load_buoy_geometry_calibrations() 

    temp_marker_buoy_geometry_config = {}
    for buoy_id_cfg, marker_ids_for_buoy_cfg in BUOY_ID_TO_MARKER_IDS_MAP.items():
        buoy_geom_cfg = {}
        buoy_id_str_cfg = str(buoy_id_cfg)

        id_X0_actual_cfg = marker_ids_for_buoy_cfg[0]
        buoy_geom_cfg[id_X0_actual_cfg] = {
            't_BRF_Xi': np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3,1),
            'R_BRF_Xi': np.eye(3, dtype=np.float32)
        }

        for i_role, marker_id_actual_cfg in enumerate(marker_ids_for_buoy_cfg):
            if i_role == 0: continue 

            marker_id_actual_str_cfg = str(marker_id_actual_cfg) 
            calibrated_pose_data = None
            use_calibrated_geometry = False

            if buoy_id_str_cfg in ALL_BUOY_GEOMETRY_CALIBRATIONS and \
               marker_id_actual_str_cfg in ALL_BUOY_GEOMETRY_CALIBRATIONS[buoy_id_str_cfg]:
                calibrated_pose_data = ALL_BUOY_GEOMETRY_CALIBRATIONS[buoy_id_str_cfg][marker_id_actual_str_cfg]
                if calibrated_pose_data and \
                   't_rel_to_X0' in calibrated_pose_data and calibrated_pose_data['t_rel_to_X0'] is not None and \
                   'rvec_rel_to_X0' in calibrated_pose_data and calibrated_pose_data['rvec_rel_to_X0'] is not None:
                    use_calibrated_geometry = True

            if use_calibrated_geometry:
                t_BRF_Xi_calib = np.array(calibrated_pose_data['t_rel_to_X0'], dtype=np.float32).reshape(3,1)
                R_BRF_Xi_calib, _ = cv2.Rodrigues(np.array(calibrated_pose_data['rvec_rel_to_X0'], dtype=np.float32))
                buoy_geom_cfg[marker_id_actual_cfg] = {
                    't_BRF_Xi': t_BRF_Xi_calib,
                    'R_BRF_Xi': R_BRF_Xi_calib
                }
            elif i_role in MARKER_RELATIVE_GEOMETRY: 
                ideal_role_geom_cfg = MARKER_RELATIVE_GEOMETRY[i_role]
                R_BRF_Xi_ideal, _ = cv2.Rodrigues(ideal_role_geom_cfg['rel_rvec_to_X0'])
                buoy_geom_cfg[marker_id_actual_cfg] = {
                    't_BRF_Xi': ideal_role_geom_cfg['rel_tvec_to_X0'].reshape(3,1),
                    'R_BRF_Xi': R_BRF_Xi_ideal
                }
            else: 
                print(f"Warning: No geometry defined for marker role {i_role} (ID {marker_id_actual_cfg}) in buoy {buoy_id_cfg}. Will be unusable.")
                buoy_geom_cfg[marker_id_actual_cfg] = {
                    't_BRF_Xi': np.full((3,1), np.nan, dtype=np.float32), 
                    'R_BRF_Xi': np.full((3,3), np.nan, dtype=np.float32)
                }
        temp_marker_buoy_geometry_config[buoy_id_cfg] = buoy_geom_cfg
    
    MARKER_BUOY_GEOMETRY_CONFIG.clear()
    MARKER_BUOY_GEOMETRY_CONFIG.update(temp_marker_buoy_geometry_config)

    BUOY_COG_OFFSETS_IN_BRF_CONFIG.clear()
    BUOY_COG_OFFSETS_IN_BRF_CONFIG.update({
        buoy_id_cfg: BUOY_COG_OFFSET_VECTOR_ASSEMBLY_FRAME.reshape(3,1) 
        for buoy_id_cfg in BUOY_ID_TO_MARKER_IDS_MAP.keys()
    })
    print("Buoy configurations initialized/re-initialized based on available calibrated or idealized geometry.")

initialize_buoy_configs()
#################################
# CORE PROCESSING FUNCTIONS
#################################

def rotation_vector_to_euler(rvec):
    # ... (no changes)
    if rvec is None:
        return np.array([[0.],[0.],[0.]]), R.identity().as_quat() 
    rvec_np = np.asarray(rvec).reshape(3) 
    if np.allclose(rvec_np, 0):
        return np.array([[0.],[0.],[0.]]), R.identity().as_quat()
    try:
        rotation = R.from_rotvec(rvec_np)
        euler_angles_rad = rotation.as_euler('xyz', degrees=False) 
        quaternion = rotation.as_quat() # [x, y, z, w] format
        return np.rad2deg(euler_angles_rad).reshape(3,1), quaternion
    except Exception as e:
        return np.array([[0.],[0.],[0.]]), R.identity().as_quat()


def transform_pose_to_reference(rvec_obj_cam, tvec_obj_cam, rvec_ref_cam, tvec_ref_cam):
    # ... (no changes)
    rvec_obj_cam_np = np.asarray(rvec_obj_cam).reshape(3,1)
    tvec_obj_cam_np = np.asarray(tvec_obj_cam).reshape(3,1)
    rvec_ref_cam_np = np.asarray(rvec_ref_cam).reshape(3,1)
    tvec_ref_cam_np = np.asarray(tvec_ref_cam).reshape(3,1)

    R_obj_cam, _ = cv2.Rodrigues(rvec_obj_cam_np)
    R_ref_cam, _ = cv2.Rodrigues(rvec_ref_cam_np)

    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R_obj_cam
    T_cam_obj[:3, 3] = tvec_obj_cam_np.flatten()

    R_ref_cam_T = R_ref_cam.T 
    t_cam_ref_inv_in_ref_coords = -R_ref_cam_T @ tvec_ref_cam_np 
    
    T_ref_cam = np.eye(4)
    T_ref_cam[:3,:3] = R_ref_cam_T
    T_ref_cam[:3,3] = t_cam_ref_inv_in_ref_coords.flatten()

    T_ref_obj = T_ref_cam @ T_cam_obj
    
    R_ref_obj = T_ref_obj[:3, :3]
    t_ref_obj_flat = T_ref_obj[:3, 3]
    rvec_ref_obj, _ = cv2.Rodrigues(R_ref_obj)
    return rvec_ref_obj.reshape(3,1), t_ref_obj_flat.reshape(3,1)

def average_quaternions_weighted(quaternions, weights):
    if not quaternions: return R.identity().as_quat()
    if len(quaternions) == 1: return quaternions[0]

    normalized_quaternions = []
    valid_weights = []
    for q, w_val in zip(quaternions, weights):
        if w_val > 1e-9: 
            norm_q = np.linalg.norm(q)
            if norm_q < 1e-9: 
                continue 
            q_norm = q / norm_q
            normalized_quaternions.append(q_norm)
            valid_weights.append(w_val)

    if not normalized_quaternions:
        return R.identity().as_quat()
    if len(normalized_quaternions) == 1:
        return normalized_quaternions[0]

    q_ref = normalized_quaternions[0]   
    for i in range(1, len(normalized_quaternions)):
        if np.dot(q_ref, normalized_quaternions[i]) < 0:
            normalized_quaternions[i] *= -1 

    M = np.zeros((4, 4))
    for q_i, w_i in zip(normalized_quaternions, valid_weights):
        q_i = q_i.reshape(4,1) 
        M += w_i * (q_i @ q_i.T) 

    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg_quat = eigenvectors[:, np.argmax(eigenvalues)]
    
    avg_quat_norm = np.linalg.norm(avg_quat)
    if avg_quat_norm < 1e-9: return R.identity().as_quat()
    return avg_quat / avg_quat_norm
    
def _process_frame_common(
    gray_frame, color_frame_to_annotate, camera_matrix, dist_coeffs, aruco_dict, aruco_parameters,
    marker_size_m, use_reference_marker_flag, reference_marker_id_val,
    positions_history, rotations_history, 
    log_date_str, log_time_str, 
    buoy_id_to_marker_ids_map_param,
    marker_buoy_geometry_config_param,
    buoy_cog_offsets_in_brf_config_param,
    log_all_marker_data_flag,
    use_kalman_filter_flag, buoy_kalman_filters_dict, dt_kf_actual 
):
    cog_log_entries = [] 
    individual_marker_log_entries = [] 
    annotated_frame = color_frame_to_annotate 
    corners, ids, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)

    ref_marker_detected_this_frame, ref_rvec, ref_tvec = False, None, None
    all_markers_data_this_frame = {} 
    rvecs_all_markers_raw, tvecs_all_markers_raw = None, None 
    
    # KF Prediction step for all existing (and initialized) filters
    if use_kalman_filter_flag:
        for buoy_id_kf_pred_loop in buoy_kalman_filters_dict.keys():
            kf_instance_loop = buoy_kalman_filters_dict[buoy_id_kf_pred_loop]
            if kf_instance_loop.initialized: # Only predict if it has been initialized
                 kf_instance_loop.predict(dt_kf_actual)
    
    if ids is not None and len(ids) > 0:
        refined_corners_list = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for corner_set in corners:
            refined_c = cv2.cornerSubPix(gray_frame, corner_set.astype(np.float32), (5,5), (-1,-1), criteria)
            refined_corners_list.append(refined_c)
        
        aruco.drawDetectedMarkers(annotated_frame, refined_corners_list, ids)
        rvecs_all_markers_raw, tvecs_all_markers_raw, _obj_points_info = aruco.estimatePoseSingleMarkers(
            refined_corners_list, marker_size_m, camera_matrix, dist_coeffs
        )

        if use_reference_marker_flag:
            for i, marker_id_array in enumerate(ids):
                if marker_id_array[0] == reference_marker_id_val:
                    ref_marker_detected_this_frame = True
                    ref_rvec, ref_tvec = rvecs_all_markers_raw[i][0].reshape(3,1), tvecs_all_markers_raw[i][0].reshape(3,1)
                    cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs, ref_rvec, ref_tvec, marker_size_m * 0.75, 3)
                    center_px_ref = tuple(np.mean(refined_corners_list[i][0], axis=0).astype(int))
                    cv2.putText(annotated_frame, "REFERENCE", (center_px_ref[0] - 40, center_px_ref[1] - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_MARKER_COLOR, 2)
                    break
        
        for i_marker, id_marker_array in enumerate(ids):
            marker_id_found = id_marker_array[0]
            r_cam_marker_raw = rvecs_all_markers_raw[i_marker][0].reshape(3,1)
            t_cam_marker_raw = tvecs_all_markers_raw[i_marker][0].reshape(3,1)
            euler_cam_marker, quat_cam_marker = rotation_vector_to_euler(r_cam_marker_raw) 
            
            marker_entry_data = {
                'cam_pos': t_cam_marker_raw.flatten(),
                'cam_rot': euler_cam_marker.flatten(), 
                'cam_quat': quat_cam_marker 
            }

            if use_reference_marker_flag and ref_marker_detected_this_frame and ref_rvec is not None and ref_tvec is not None:
                try:
                    r_rel_marker, t_rel_marker = transform_pose_to_reference(
                        r_cam_marker_raw, t_cam_marker_raw, ref_rvec, ref_tvec)
                    euler_rel_marker, quat_rel_marker = rotation_vector_to_euler(r_rel_marker) 
                    marker_entry_data['rel_pos'] = t_rel_marker.flatten()
                    marker_entry_data['rel_rot'] = euler_rel_marker.flatten() 
                    marker_entry_data['rel_quat'] = quat_rel_marker 
                except Exception: pass 
            all_markers_data_this_frame[marker_id_found] = marker_entry_data
        
        if log_all_marker_data_flag:
            for marker_id_logged, data_for_log in all_markers_data_this_frame.items():
                marker_log_row = [log_date_str, log_time_str, marker_id_logged] 
                marker_log_row.extend([f"{v:.6f}" for v in data_for_log['cam_pos']])
                marker_log_row.extend([f"{v:.3f}" for v in data_for_log['cam_rot']])
                marker_log_row.extend([f"{v:.6f}" for v in data_for_log.get('cam_quat', [np.nan]*4)])
                
                if use_reference_marker_flag:
                    if 'rel_pos' in data_for_log and 'rel_rot' in data_for_log:
                        marker_log_row.extend([f"{v:.6f}" for v in data_for_log['rel_pos']])
                        marker_log_row.extend([f"{v:.3f}" for v in data_for_log['rel_rot']])
                        marker_log_row.extend([f"{v:.6f}" for v in data_for_log.get('rel_quat', [np.nan]*4)])
                    else:
                        marker_log_row.extend(["NaN"] * (3 + 3 + 4)) 
                individual_marker_log_entries.append(marker_log_row)

        detected_markers_for_buoy_fusion = {} 
        marker_object_points_generic = np.array([
            [-marker_size_m/2,  marker_size_m/2, 0], [ marker_size_m/2,  marker_size_m/2, 0],
            [ marker_size_m/2, -marker_size_m/2, 0], [-marker_size_m/2, -marker_size_m/2, 0]
        ], dtype=np.float32)

        for i, marker_id_array in enumerate(ids):
            marker_id = marker_id_array[0]
            r_cam_Xi_raw, t_cam_Xi_raw = rvecs_all_markers_raw[i][0].reshape(3,1), tvecs_all_markers_raw[i][0].reshape(3,1)
            
            for buoy_id_loop, defined_marker_ids_for_buoy in buoy_id_to_marker_ids_map_param.items():
                if marker_id in defined_marker_ids_for_buoy:
                    buoy_specific_marker_geometry_map = marker_buoy_geometry_config_param.get(buoy_id_loop)
                    if not buoy_specific_marker_geometry_map or marker_id not in buoy_specific_marker_geometry_map: continue
                    geom_Xi_in_BRF = buoy_specific_marker_geometry_map[marker_id]
                    if np.isnan(geom_Xi_in_BRF['t_BRF_Xi']).any() or np.isnan(geom_Xi_in_BRF['R_BRF_Xi']).any(): continue 

                    t_BRF_Xi, R_BRF_Xi = geom_Xi_in_BRF['t_BRF_Xi'], geom_Xi_in_BRF['R_BRF_Xi']
                    R_cam_Xi, _ = cv2.Rodrigues(r_cam_Xi_raw)
                    R_Xi_BRF = R_BRF_Xi.T; t_Xi_BRF = -R_Xi_BRF @ t_BRF_Xi 
                    R_cam_BRF_est_from_Xi = R_cam_Xi @ R_Xi_BRF
                    t_cam_BRF_est_from_Xi = (R_cam_Xi @ t_Xi_BRF) + t_cam_Xi_raw 
                    
                    projected_corners, _ = cv2.projectPoints(marker_object_points_generic, r_cam_Xi_raw, t_cam_Xi_raw, camera_matrix, dist_coeffs)
                    reprojection_error = cv2.norm(refined_corners_list[i].reshape(-1,2), projected_corners.reshape(-1,2), cv2.NORM_L2)
                    confidence_Xi = 1.0 / (1.0 + reprojection_error**2 + 1e-9) 

                    detected_markers_for_buoy_fusion.setdefault(buoy_id_loop, []).append({
                        'R_matrix_cam_BRF_est': R_cam_BRF_est_from_Xi, 
                        'tvec_cam_BRF_est': t_cam_BRF_est_from_Xi.reshape(3,1),
                        'confidence': confidence_Xi, 'source_marker_id': marker_id
                    })
                    break 

        for buoy_id, estimates_for_this_buoy in detected_markers_for_buoy_fusion.items():
            if not estimates_for_this_buoy: continue
            
            final_R_cam_BRF_matrix = None
            final_tvec_cam_BRF = None 
            quats_to_average, weights_for_rot_avg, tvecs_to_average, weights_for_tvec_avg = [], [], [], []
            for est in estimates_for_this_buoy:
                if est['R_matrix_cam_BRF_est'] is not None:
                    try:
                        quats_to_average.append(R.from_matrix(est['R_matrix_cam_BRF_est']).as_quat())
                        weights_for_rot_avg.append(est['confidence'])
                    except (cv2.error, ValueError): pass
                if est['tvec_cam_BRF_est'] is not None:
                    tvecs_to_average.append(est['tvec_cam_BRF_est'])
                    weights_for_tvec_avg.append(est['confidence'])
            if quats_to_average:
                try:
                    avg_quat = average_quaternions_weighted(quats_to_average, weights_for_rot_avg)
                    final_R_cam_BRF_matrix = R.from_quat(avg_quat).as_matrix()
                except Exception: final_R_cam_BRF_matrix = None
            if tvecs_to_average:
                tvecs_np = np.hstack(tvecs_to_average) 
                weights_np = np.array(weights_for_tvec_avg).reshape(1, -1) 
                sum_weights = np.sum(weights_np)
                if sum_weights > 1e-6:
                    final_tvec_cam_BRF = (np.sum(tvecs_np * weights_np, axis=1) / sum_weights).reshape(3,1)
                elif tvecs_to_average: 
                    final_tvec_cam_BRF = np.mean(tvecs_np, axis=1).reshape(3,1)

            if final_R_cam_BRF_matrix is None or final_tvec_cam_BRF is None: continue
            
            rvec_cam_BRF_final, _ = cv2.Rodrigues(final_R_cam_BRF_matrix)
            buoy_cog_offset_in_brf_coords = buoy_cog_offsets_in_brf_config_param.get(buoy_id, np.zeros((3,1)))
            
            # Raw CoG pose in camera coordinates
            rvec_cog_cam_raw = rvec_cam_BRF_final 
            tvec_cog_cam_raw = (final_R_cam_BRF_matrix @ buoy_cog_offset_in_brf_coords) + final_tvec_cam_BRF
            euler_angles_cog_cam_raw_deg, quat_cog_cam_raw = rotation_vector_to_euler(rvec_cog_cam_raw)

            # Initialize working CoG pose variables with raw data
            rvec_cog_cam = rvec_cog_cam_raw.copy()
            tvec_cog_cam = tvec_cog_cam_raw.copy()
            euler_angles_cog_cam_deg = euler_angles_cog_cam_raw_deg.copy()
            quat_cog_cam = quat_cog_cam_raw.copy()

            # Apply Kalman Filter if enabled
            if use_kalman_filter_flag:
                if buoy_id not in buoy_kalman_filters_dict:
                    buoy_kalman_filters_dict[buoy_id] = KalmanFilterPose6DOF_Quat(
                        process_noise_vel_pos=KF_PROCESS_NOISE_VEL_POS,
                        process_noise_vel_ori=KF_PROCESS_NOISE_VEL_ORI,
                        measurement_noise_pos=KF_MEASUREMENT_NOISE_POS,
                        measurement_noise_ori=KF_MEASUREMENT_NOISE_ORI
                    )
                
                kf = buoy_kalman_filters_dict[buoy_id]
                
                # Measurement vector z: [tx, ty, tz, qx, qy, qz, qw]
                measurement_pos_cam_raw_flat = tvec_cog_cam_raw.flatten() 
                measurement_quat_cam_raw_flat = quat_cog_cam_raw.flatten() 
                z_k_cam_pose = np.concatenate((measurement_pos_cam_raw_flat, measurement_quat_cam_raw_flat))
                
                # Predict was called for initialized KFs at the start of the function.
                # Update will initialize if it's the first time for this KF.
                filtered_state_cam_pose = kf.update(z_k_cam_pose) 
                
                tvec_cog_cam = filtered_state_cam_pose[0:3].reshape(3,1)
                quat_cog_cam = filtered_state_cam_pose[3:7] # This is [x,y,z,w]
                
                try:
                    rot_obj_filtered_cam = R.from_quat(quat_cog_cam)
                    rvec_cog_cam = rot_obj_filtered_cam.as_rotvec().reshape(3,1)
                    euler_angles_cog_cam_deg_flat = rot_obj_filtered_cam.as_euler('xyz', degrees=True)
                    euler_angles_cog_cam_deg = euler_angles_cog_cam_deg_flat.reshape(3,1)
                except Exception as e:
                    print(f"Error converting filtered quaternion to rvec/euler for buoy {buoy_id}: {e}")
                    rvec_cog_cam = rvec_cog_cam_raw.copy()
                    euler_angles_cog_cam_deg = euler_angles_cog_cam_raw_deg.copy()
                    quat_cog_cam = quat_cog_cam_raw.copy()

            current_log_rvec_cog = rvec_cog_cam 
            current_log_tvec_cog = tvec_cog_cam 
            is_cog_pose_relative = False 

            if use_reference_marker_flag and ref_marker_detected_this_frame and ref_rvec is not None and ref_tvec is not None:
                try:
                    rvec_cog_relative, tvec_cog_relative = transform_pose_to_reference(
                        rvec_cog_cam, tvec_cog_cam, ref_rvec, ref_tvec)
                    current_log_rvec_cog = rvec_cog_relative
                    current_log_tvec_cog = tvec_cog_relative
                    is_cog_pose_relative = True
                except Exception as e:
                    print(f"Error transforming filtered CoG to relative frame for buoy {buoy_id}: {e}")
                    is_cog_pose_relative = False
            
            euler_angles_cog_for_log_deg, quat_cog_for_log = rotation_vector_to_euler(current_log_rvec_cog)

            cog_log_entry_data = [log_date_str, log_time_str, buoy_id]
            if use_reference_marker_flag: 
                cog_log_entry_data.append("1" if ref_marker_detected_this_frame and is_cog_pose_relative else "0") 
                if is_cog_pose_relative: 
                    cog_log_entry_data.extend([f"{v:.6f}" for v in current_log_tvec_cog.flatten()])
                    cog_log_entry_data.extend([f"{val[0]:.3f}" for val in euler_angles_cog_for_log_deg])
                    cog_log_entry_data.extend([f"{v:.6f}" for v in quat_cog_for_log]) 
                else: 
                    cog_log_entry_data.extend(["NaN"] * (3 + 3 + 4))
                cog_log_entry_data.extend([f"{v:.6f}" for v in tvec_cog_cam.flatten()]) 
                cog_log_entry_data.extend([f"{val[0]:.3f}" for val in euler_angles_cog_cam_deg])
                cog_log_entry_data.extend([f"{v:.6f}" for v in quat_cog_cam])
            else: 
                cog_log_entry_data.extend([f"{v:.6f}" for v in tvec_cog_cam.flatten()]) 
                cog_log_entry_data.extend([f"{val[0]:.3f}" for val in euler_angles_cog_cam_deg])
                cog_log_entry_data.extend([f"{v:.6f}" for v in quat_cog_cam])
            cog_log_entries.append(cog_log_entry_data)

            try:
                imgpts_cog_origin, _ = cv2.projectPoints(np.array([[[0.,0.,0.]]], dtype=np.float32), 
                                                         rvec_cog_cam, tvec_cog_cam, camera_matrix, dist_coeffs)
                center_px_cog_on_image = tuple(imgpts_cog_origin[0,0,:].astype(int))
                if buoy_id not in positions_history: positions_history[buoy_id] = []
                if buoy_id not in rotations_history: rotations_history[buoy_id] = [] 
                positions_history[buoy_id].append(center_px_cog_on_image)
                rotations_history[buoy_id].append(euler_angles_cog_for_log_deg)
                current_trajectory_length = TRAJECTORY_LENGTH 
                if len(positions_history[buoy_id]) > current_trajectory_length:
                    positions_history[buoy_id].pop(0)
                    rotations_history[buoy_id].pop(0)

                cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs, rvec_cog_cam, tvec_cog_cam, marker_size_m * 0.6, 3)
                prefix_text = ("Rel." if is_cog_pose_relative else "Abs.") + (" (F)" if use_kalman_filter_flag else "")
                y_off = -40
                cv2.putText(annotated_frame, f"B{buoy_id} CoG", (center_px_cog_on_image[0]+10, center_px_cog_on_image[1]+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BUOY_COG_INFO_COLOR, 2); y_off+=20
                cv2.putText(annotated_frame, f"{prefix_text}P:({current_log_tvec_cog[0,0]:.2f},{current_log_tvec_cog[1,0]:.2f},{current_log_tvec_cog[2,0]:.2f})", (center_px_cog_on_image[0]+10, center_px_cog_on_image[1]+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BUOY_COG_INFO_COLOR, 1); y_off+=15 
                cv2.putText(annotated_frame, f"{prefix_text}R:({euler_angles_cog_for_log_deg[0,0]:.1f},{euler_angles_cog_for_log_deg[1,0]:.1f},{euler_angles_cog_for_log_deg[2,0]:.1f})", (center_px_cog_on_image[0]+10, center_px_cog_on_image[1]+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BUOY_COG_INFO_COLOR, 1)
            except Exception as e:
                print(f"Error during CoG visualization for buoy {buoy_id}: {e}")
                pass 

    for buoy_id_hist, pos_list_hist in positions_history.items():
        for i in range(1, len(pos_list_hist)):
            if pos_list_hist[i-1] is not None and pos_list_hist[i] is not None: 
                try:
                    pt1, pt2 = tuple(map(int, pos_list_hist[i-1])), tuple(map(int, pos_list_hist[i]))
                    cv2.line(annotated_frame, pt1, pt2, TRAJECTORY_COLOR, 2)
                except Exception: pass
    
    return annotated_frame, cog_log_entries, individual_marker_log_entries, positions_history, rotations_history, ref_marker_detected_this_frame

#################################
# BACKEND APPLICATION FUNCTIONS
#################################
def track_buoy_with_aruco_6dof(parent_tk_window=None):
    # ... (no changes from previous KF version, except buoy_kalman_filters_live will store KalmanFilterPose6DOF_Quat instances)
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_live_{experiment_timestamp}"
    experiment_dir = os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created experiment directory: {experiment_dir}")

    cog_csv_fn = os.path.join(experiment_dir, "tracking_data_cog.csv")
    markers_csv_fn = os.path.join(experiment_dir, "tracking_data_markers.csv")
    annotated_video_filename_live = f"live_annotated_{experiment_timestamp}{VIDEO_EXTENSION}" 
    full_annotated_video_path_live = os.path.join(experiment_dir, annotated_video_filename_live)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
    if not cap.isOpened(): messagebox.showerror("Cam Err", f"No cam {CAMERA_INDEX}", parent=parent_tk_window); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1]); cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    w, h, fps_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    print(f"Cam opened. Actual: {w}x{h} @ {fps_actual:.2f}FPS.")

    cam_mat, dist_c = load_camera_calibration()
    aruco_dict_obj, params_obj = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE), aruco.DetectorParameters()
    pos_live, rot_live = {}, {}
    
    buoy_kalman_filters_live = {} # Will store KalmanFilterPose6DOF_Quat instances
    last_kf_processing_time_live = None

    header_cog = ['Date', 'Time', 'Buoy_ID']
    # ... (header logic is the same)
    if USE_REFERENCE_MARKER: 
        header_cog.extend(['Ref_Marker_Frame_Detected', 
                       'CoG_Pos_X_Rel', 'CoG_Pos_Y_Rel', 'CoG_Pos_Z_Rel', 
                       'CoG_Rot_X_Rel_deg', 'CoG_Rot_Y_Rel_deg', 'CoG_Rot_Z_Rel_deg',
                       'CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel',
                       'CoG_Pos_X_Cam', 'CoG_Pos_Y_Cam', 'CoG_Pos_Z_Cam', 
                       'CoG_Rot_X_Cam_deg', 'CoG_Rot_Y_Cam_deg', 'CoG_Rot_Z_Cam_deg',
                       'CoG_Quat_X_Cam', 'CoG_Quat_Y_Cam', 'CoG_Quat_Z_Cam', 'CoG_Quat_W_Cam'])
    else: 
        header_cog.extend(['CoG_Pos_X_Cam', 'CoG_Pos_Y_Cam', 'CoG_Pos_Z_Cam', 
                       'CoG_Rot_X_Cam_deg', 'CoG_Rot_Y_Cam_deg', 'CoG_Rot_Z_Cam_deg',
                       'CoG_Quat_X_Cam', 'CoG_Quat_Y_Cam', 'CoG_Quat_Z_Cam', 'CoG_Quat_W_Cam'])
    
    with open(cog_csv_fn, 'w', newline='') as f_cog: csv.writer(f_cog).writerow(header_cog)
    print(f"CoG CSV file created: {cog_csv_fn}")

    header_markers = None
    if LOG_ALL_MARKER_DATA:
        # ... (header logic is the same)
        header_markers = ['Log_Date', 'Log_Time', 'Marker_ID',
                          'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z',
                          'Cam_Rot_X_deg', 'Cam_Rot_Y_deg', 'Cam_Rot_Z_deg',
                          'Cam_Quat_X', 'Cam_Quat_Y', 'Cam_Quat_Z', 'Cam_Quat_W']
        if USE_REFERENCE_MARKER:
            header_markers.extend(['Rel_Pos_X', 'Rel_Pos_Y', 'Rel_Pos_Z',
                                   'Rel_Rot_X_deg', 'Rel_Rot_Y_deg', 'Rel_Rot_Z_deg',
                                   'Rel_Quat_X', 'Rel_Quat_Y', 'Rel_Quat_Z', 'Rel_Quat_W'])
        with open(markers_csv_fn, 'w', newline='') as f_markers: csv.writer(f_markers).writerow(header_markers)
        print(f"Markers CSV file created: {markers_csv_fn}")
    
    rec_vid_flag, vid_writer = RECORD_VIDEO, None
    if rec_vid_flag: 
        vid_writer = get_video_writer((w,h), full_annotated_video_path_live, fps=fps_actual if fps_actual > 0 else VIDEO_FPS)
        if not vid_writer: 
            rec_vid_flag = False; 
            messagebox.showwarning("Rec Err", f"Could not create video writer for {full_annotated_video_path_live}. Recording disabled.", parent=parent_tk_window)
        else:
            print(f"Video recording will start to: {full_annotated_video_path_live}")

    rev_disp, ref_detected_stat = REVERSE_CAMERA_DISPLAY, False
    current_video_file_being_written = full_annotated_video_path_live if rec_vid_flag and vid_writer else None

    cog_csv_file_obj = open(cog_csv_fn, 'a', newline='')
    cog_csv_writer_obj = csv.writer(cog_csv_file_obj)
    
    markers_csv_file_obj, markers_csv_writer_obj = None, None
    if LOG_ALL_MARKER_DATA:
        markers_csv_file_obj = open(markers_csv_fn, 'a', newline='')
        markers_csv_writer_obj = csv.writer(markers_csv_file_obj)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: messagebox.showerror("Cam Error", "Failed to grab frame.", parent=parent_tk_window); break
            
            current_frame_time = time.time() 
            if last_kf_processing_time_live is None: 
                last_kf_processing_time_live = current_frame_time - (1.0 / (fps_actual if fps_actual > 0 else CAMERA_FPS)) 

            dt_kf_actual = current_frame_time - last_kf_processing_time_live
            if dt_kf_actual <= 1e-6: # Ensure dt is positive and sensible
                dt_kf_actual = (1.0 / (fps_actual if fps_actual > 0 else CAMERA_FPS)) # Fallback

            frame_copy_for_processing = frame.copy()
            gray = cv2.cvtColor(frame_copy_for_processing, cv2.COLOR_BGR2GRAY)
            now_dt_obj = datetime.now(); date_s, time_s = now_dt_obj.strftime("%Y-%m-%d"), now_dt_obj.strftime("%H:%M:%S.%f")[:-3]

            ann_fr, cog_log_entries_list, marker_log_entries_list, pos_live, rot_live, ref_detected_stat = _process_frame_common(
                gray, frame_copy_for_processing, cam_mat, dist_c, aruco_dict_obj, params_obj, MARKER_SIZE_METERS,
                USE_REFERENCE_MARKER, REFERENCE_MARKER_ID, 
                pos_live, rot_live,
                date_s, time_s, 
                BUOY_ID_TO_MARKER_IDS_MAP, 
                MARKER_BUOY_GEOMETRY_CONFIG, 
                BUOY_COG_OFFSETS_IN_BRF_CONFIG,
                LOG_ALL_MARKER_DATA,
                USE_KALMAN_FILTER, buoy_kalman_filters_live, dt_kf_actual
                )
            
            last_kf_processing_time_live = current_frame_time 
            
            if cog_log_entries_list:
                cog_csv_writer_obj.writerows(cog_log_entries_list)
            if LOG_ALL_MARKER_DATA and marker_log_entries_list and markers_csv_writer_obj:
                markers_csv_writer_obj.writerows(marker_log_entries_list)

            if USE_REFERENCE_MARKER: 
                ref_txt = f"Ref.ID {REFERENCE_MARKER_ID}: {'DET' if ref_detected_stat else 'NOT DET'}"
                cv2.putText(ann_fr, ref_txt, (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, REFERENCE_MARKER_COLOR if ref_detected_stat else (0,0,255), 2)
            cv2.putText(ann_fr, f"RevDisp: {'ON' if rev_disp else 'OFF'}(r)", (10,h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255),2)
            cv2.putText(ann_fr, f"{date_s} {time_s}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TIMESTAMP_COLOR, 2)
            
            rec_txt_y = h-80
            if rec_vid_flag and vid_writer: 
                cv2.circle(ann_fr, (w-30,30), 10, RECORD_INDICATOR_COLOR, -1); cv2.putText(ann_fr, "REC", (w-75,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RECORD_INDICATOR_COLOR, 2)
                cv2.putText(ann_fr, f"Rec: ON(v)", (10,rec_txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RECORD_INDICATOR_COLOR, 2)
            else: cv2.putText(ann_fr, f"Rec: OFF(v)", (10,rec_txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,128,128), 2)

            disp_fr = cv2.flip(ann_fr, 1) if rev_disp else ann_fr
            if rec_vid_flag and vid_writer: vid_writer.write(ann_fr)
            cv2.imshow('Buoy 6DOF Tracking (Live) - Q to Exit', disp_fr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): rev_disp = not rev_disp
            elif key == ord('v'):
                rec_vid_flag = not rec_vid_flag
                if rec_vid_flag: 
                    if not vid_writer: 
                        vid_writer = get_video_writer((w,h), full_annotated_video_path_live, fps=fps_actual if fps_actual > 0 else VIDEO_FPS)
                        if not vid_writer: 
                            rec_vid_flag = False 
                            messagebox.showwarning("Rec Err", f"Failed to start video writer for {full_annotated_video_path_live}. Recording remains OFF.", parent=parent_tk_window)
                        else:
                            current_video_file_being_written = full_annotated_video_path_live
                            print(f"Recording started: {current_video_file_being_written}")
                elif vid_writer: 
                    vid_writer.release()
                    print(f"Video saved: {current_video_file_being_written}")
                    vid_writer = None 
                    current_video_file_being_written = None
    finally:
        if cog_csv_file_obj: cog_csv_file_obj.close()
        if markers_csv_file_obj: markers_csv_file_obj.close()

    if vid_writer: 
        vid_writer.release()
        if current_video_file_being_written:
            print(f"Video recording stopped. Finalized: {current_video_file_being_written}")
    cap.release(); cv2.destroyAllWindows()
    
    saved_files_msg = f"CoG data in: {cog_csv_fn}"
    if LOG_ALL_MARKER_DATA: saved_files_msg += f"\nMarkers data in: {markers_csv_fn}"
    messagebox.showinfo("Live Tracking Ended", f"Experiment data saved in experiment folder:\n{experiment_dir}\n\n{saved_files_msg}", parent=parent_tk_window)

def process_video_offline(input_video_path, use_ref_marker_override,
                          gui_status_label=None, gui_progress_bar=None,
                          video_start_date_str=None, video_start_time_str=None): 
    # ... (no changes from previous KF version, except buoy_kalman_filters_offline will store KalmanFilterPose6DOF_Quat instances)
    parent_tk = gui_status_label.master if gui_status_label else (gui_progress_bar.master if gui_progress_bar else None)
    if not os.path.exists(input_video_path): messagebox.showerror("File Err", f"No vid: {input_video_path}", parent=parent_tk); return

    input_video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_offline_{input_video_basename}_{experiment_timestamp}"
    experiment_dir = os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created experiment directory: {experiment_dir}")

    original_video_filename = f"original_{os.path.basename(input_video_path)}"
    original_video_dst_path = os.path.join(experiment_dir, original_video_filename)
    try:
        shutil.copy2(input_video_path, original_video_dst_path)
        print(f"Copied input video to: {original_video_dst_path}")
    except Exception as e:
        print(f"Warning: Could not copy input video to experiment folder: {e}")
        messagebox.showwarning("File Copy Warning", f"Could not copy input video to experiment folder: {e}\nProcessing will continue with the original path.", parent=parent_tk)

    cog_csv_fn = os.path.join(experiment_dir, "tracking_data_cog.csv")
    markers_csv_fn = os.path.join(experiment_dir, "tracking_data_markers.csv")
    annotated_video_filename = f"annotated_video_{experiment_timestamp}{VIDEO_EXTENSION}"
    full_annotated_video_path = os.path.join(experiment_dir, annotated_video_filename)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened(): messagebox.showerror("Vid Err", f"No open: {input_video_path}", parent=parent_tk); return

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS); fps_orig = fps_orig if fps_orig > 0 else VIDEO_FPS
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if gui_status_label: gui_status_label.config(text=f"Proc: {os.path.basename(input_video_path)} (0%)")
    if gui_progress_bar: gui_progress_bar["value"]=0; gui_progress_bar["maximum"]= total_fr if total_fr > 0 else 100

    cam_mat, dist_c = load_camera_calibration()
    aruco_dict_obj, params_obj = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE), aruco.DetectorParameters()
    pos_off, rot_off = {}, {}

    buoy_kalman_filters_offline = {} # Will store KalmanFilterPose6DOF_Quat instances
    last_kf_processing_time_offline_sec = None 

    ann_vid_writer = get_video_writer((w,h), full_annotated_video_path, fps=fps_orig)
    if not ann_vid_writer:
        messagebox.showwarning("Vid Writer Err",f"Failed to create annotated video writer for {full_annotated_video_path}. Annotated video will not be saved.", parent=parent_tk)
    else:
        print(f"Annotated video will be saved to: {full_annotated_video_path}")

    header_cog = ['Date','Time','Buoy_ID']
    # ... (header logic is the same)
    if use_ref_marker_override:
        header_cog.extend(['Ref_Marker_Frame_Detected',
                       'CoG_Pos_X_Rel','CoG_Pos_Y_Rel','CoG_Pos_Z_Rel',
                       'CoG_Rot_X_Rel_deg','CoG_Rot_Y_Rel_deg','CoG_Rot_Z_Rel_deg',
                       'CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel', 
                       'CoG_Pos_X_Cam','CoG_Pos_Y_Cam','CoG_Pos_Z_Cam',
                       'CoG_Rot_X_Cam_deg','CoG_Rot_Y_Cam_deg','CoG_Rot_Z_Cam_deg',
                       'CoG_Quat_X_Cam', 'CoG_Quat_Y_Cam', 'CoG_Quat_Z_Cam', 'CoG_Quat_W_Cam'])
    else:
        header_cog.extend(['CoG_Pos_X_Cam','CoG_Pos_Y_Cam','CoG_Pos_Z_Cam',
                       'CoG_Rot_X_Cam_deg','CoG_Rot_Y_Cam_deg','CoG_Rot_Z_Cam_deg',
                       'CoG_Quat_X_Cam', 'CoG_Quat_Y_Cam', 'CoG_Quat_Z_Cam', 'CoG_Quat_W_Cam'])
    with open(cog_csv_fn, 'w', newline='') as f_cog: csv.writer(f_cog).writerow(header_cog)
    print(f"CoG CSV file created: {cog_csv_fn}")

    header_markers = None
    if LOG_ALL_MARKER_DATA:
        # ... (header logic is the same)
        header_markers = ['Log_Date', 'Log_Time', 'Marker_ID',
                          'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z',
                          'Cam_Rot_X_deg', 'Cam_Rot_Y_deg', 'Cam_Rot_Z_deg',
                          'Cam_Quat_X', 'Cam_Quat_Y', 'Cam_Quat_Z', 'Cam_Quat_W']
        if use_ref_marker_override:
            header_markers.extend(['Rel_Pos_X', 'Rel_Pos_Y', 'Rel_Pos_Z',
                                   'Rel_Rot_X_deg', 'Rel_Rot_Y_deg', 'Rel_Rot_Z_deg',
                                   'Rel_Quat_X', 'Rel_Quat_Y', 'Rel_Quat_Z', 'Rel_Quat_W'])
        with open(markers_csv_fn, 'w', newline='') as f_markers: csv.writer(f_markers).writerow(header_markers)
        print(f"Markers CSV file created: {markers_csv_fn}")

    base_timestamp_for_video = None
    timestamp_info_msg = ""
    if video_start_date_str and video_start_time_str:
        try:
            full_timestamp_str = f"{video_start_date_str.strip()} {video_start_time_str.strip()}"
            base_timestamp_for_video = datetime.strptime(full_timestamp_str, "%Y-%m-%d %H:%M:%S")
            timestamp_info_msg = f"Using user-provided video start timestamp: {base_timestamp_for_video}"
            print(timestamp_info_msg)
        except ValueError:
            warning_msg = (f"Warning: Could not parse provided date/time: '{video_start_date_str} {video_start_time_str}'.\n"
                           f"Please use YYYY-MM-DD and HH:MM:SS format.\n"
                           f"Falling back to processing date and relative video time.")
            print(warning_msg)
            messagebox.showwarning("Date/Time Parse Error", warning_msg, parent=parent_tk)
            timestamp_info_msg = "Using processing date and relative video time due to parse error of user input."
    else:
        timestamp_info_msg = "No video start date/time provided by user. Using processing date and relative video time."
        print(timestamp_info_msg)

    print(f"Offline processing. Ref Marker Logic for this run: {'Override ON (using Ref Marker)' if use_ref_marker_override else 'Override OFF (ignoring Ref Marker)'}.")
    print(f"Logging all marker data: {'Enabled (separate markers CSV)' if LOG_ALL_MARKER_DATA else 'Disabled'}")

    fr_count, start_t_proc, ref_det_offline = 0, time.time(), False

    cog_csv_file_obj = open(cog_csv_fn, 'a', newline='')
    cog_csv_writer_obj = csv.writer(cog_csv_file_obj)
    markers_csv_file_obj, markers_csv_writer_obj = None, None
    if LOG_ALL_MARKER_DATA:
        markers_csv_file_obj = open(markers_csv_fn, 'a', newline='')
        markers_csv_writer_obj = csv.writer(markers_csv_file_obj)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            fr_count += 1
            frame_copy_for_processing = frame.copy() # For processing

            current_frame_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if current_frame_time_msec <= 0 and fr_count > 1 and last_kf_processing_time_offline_sec is not None:
                current_frame_time_msec = (last_kf_processing_time_offline_sec * 1000.0) + (1000.0 / fps_orig if fps_orig > 0 else 1000.0/30.0)
            elif current_frame_time_msec < 0 : # Ensure non-negative
                 current_frame_time_msec = (fr_count -1) * (1000.0 / fps_orig if fps_orig > 0 else 1000.0/30.0)


            current_frame_time_sec = current_frame_time_msec / 1000.0

            if last_kf_processing_time_offline_sec is None: 
                 last_kf_processing_time_offline_sec = current_frame_time_sec - (1.0 / fps_orig if fps_orig > 0 else 1.0/30.0)

            dt_kf_actual = current_frame_time_sec - last_kf_processing_time_offline_sec
            if dt_kf_actual <= 1e-6 : 
                 dt_kf_actual = (1.0 / fps_orig if fps_orig > 0 else 1.0/30.0) 
            
            time_offset_from_video_start = timedelta(milliseconds=current_frame_time_msec)
            current_log_date_str, current_log_time_str = "", ""

            if base_timestamp_for_video:
                current_frame_datetime = base_timestamp_for_video + time_offset_from_video_start
                current_log_date_str = current_frame_datetime.strftime("%Y-%m-%d")
                current_log_time_str = current_frame_datetime.strftime("%H:%M:%S.%f")[:-3]
            else:
                current_log_date_str = datetime.now().strftime("%Y-%m-%d") 
                relative_time_obj = datetime.min + time_offset_from_video_start 
                current_log_time_str = relative_time_obj.strftime("%H:%M:%S.%f")[:-3]

            gray = cv2.cvtColor(frame_copy_for_processing, cv2.COLOR_BGR2GRAY)

            ann_fr, cog_log_entries_list, marker_log_entries_list, pos_off, rot_off, ref_det_offline = _process_frame_common(
                gray, frame_copy_for_processing, cam_mat, dist_c, aruco_dict_obj, params_obj, MARKER_SIZE_METERS,
                use_ref_marker_override, REFERENCE_MARKER_ID,
                pos_off, rot_off, 
                current_log_date_str, current_log_time_str, 
                BUOY_ID_TO_MARKER_IDS_MAP,
                MARKER_BUOY_GEOMETRY_CONFIG,
                BUOY_COG_OFFSETS_IN_BRF_CONFIG,
                LOG_ALL_MARKER_DATA,
                USE_KALMAN_FILTER, buoy_kalman_filters_offline, dt_kf_actual
                )
            
            last_kf_processing_time_offline_sec = current_frame_time_sec 
            
            if cog_log_entries_list:
                cog_csv_writer_obj.writerows(cog_log_entries_list)
            if LOG_ALL_MARKER_DATA and marker_log_entries_list and markers_csv_writer_obj:
                markers_csv_writer_obj.writerows(marker_log_entries_list)

            if use_ref_marker_override:
                ref_txt = f"Ref.ID {REFERENCE_MARKER_ID}: {'DET' if ref_det_offline else 'NOT DET'}"
                cv2.putText(ann_fr, ref_txt, (10,h-20), cv2.FONT_HERSHEY_SIMPLEX,0.7,REFERENCE_MARKER_COLOR if ref_det_offline else (0,0,255),2)
            
            cv2.putText(ann_fr, f"{current_log_date_str} {current_log_time_str}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,TIMESTAMP_COLOR,2)
            prog_txt = f"Frame: {fr_count}" + (f"/{total_fr}" if total_fr > 0 else "")
            cv2.putText(ann_fr, prog_txt, (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,OFFLINE_PROGRESS_COLOR,2)

            if ann_vid_writer: ann_vid_writer.write(ann_fr)
            if SHOW_OFFLINE_PROCESSING_PREVIEW:
                if w*OFFLINE_PREVIEW_RESIZE_FACTOR > 0 and h*OFFLINE_PREVIEW_RESIZE_FACTOR > 0 :
                    try:
                        disp_res = cv2.resize(ann_fr, (int(w*OFFLINE_PREVIEW_RESIZE_FACTOR), int(h*OFFLINE_PREVIEW_RESIZE_FACTOR)))
                        cv2.imshow('Offline Preview - Q to Stop Early', disp_res)
                        if (cv2.waitKey(1)&0xFF)==ord('q'): print("Offline processing interrupted by user."); break
                    except cv2.error:
                        cv2.imshow('Offline Preview - Q to Stop Early', ann_fr)
                        if (cv2.waitKey(1)&0xFF)==ord('q'): print("Offline processing interrupted by user."); break

            if total_fr > 0 and (fr_count % 50 == 0 or fr_count == total_fr):
                el_t = time.time()-start_t_proc; fps_p = fr_count/el_t if el_t > 0 else 0; prog_p = (fr_count/total_fr)*100
                print(f"Processed {fr_count}/{total_fr} ({fps_p:.2f} fps), {prog_p:.1f}%")
                if gui_status_label: gui_status_label.config(text=f"Proc: {os.path.basename(input_video_path)} ({prog_p:.1f}%)")
                if gui_progress_bar: gui_progress_bar["value"] = fr_count
                if parent_tk: parent_tk.update_idletasks()
    finally:
        if cog_csv_file_obj: cog_csv_file_obj.close()
        if markers_csv_file_obj: markers_csv_file_obj.close()

    cap.release()
    if ann_vid_writer: ann_vid_writer.release(); print(f"Annotated video saved: {full_annotated_video_path}")
    if SHOW_OFFLINE_PROCESSING_PREVIEW: cv2.destroyAllWindows()

    saved_files_msg = f"CoG data in: {cog_csv_fn}"
    if LOG_ALL_MARKER_DATA: saved_files_msg += f"\nMarkers data in: {markers_csv_fn}"
    final_msg = (f"Offline processing complete. Data saved in experiment folder:\n{experiment_dir}\n\n"
                 f"{saved_files_msg}\n\nTimestamping note: {timestamp_info_msg}")

    if gui_status_label: gui_status_label.config(text=f"Done. Data in: {experiment_name}")
    messagebox.showinfo("Processing Complete", final_msg, parent=parent_tk)

def calibrate_camera(parent_tk_window=None):
    # ... (no changes)
    print("Starting Camera Calibration process...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
    if not cap.isOpened(): 
        messagebox.showerror("Camera Error", f"Cannot open camera {CAMERA_INDEX} for calibration.", parent=parent_tk_window)
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    
    rev_disp_calib = REVERSE_CAMERA_DISPLAY
    chessboard_shape = CALIBRATION_CHESSBOARD_SHAPE
    square_size_meters = CHESSBOARD_SQUARE_SIZE_MM / 1000.0

    objp = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2) * square_size_meters

    obj_points_list, img_points_list = [], []
    gray_frame_shape = None

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
        
        cv2.putText(display_frame_calib, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(display_frame_calib, f"Captured: {len(obj_points_list)}", (10, display_frame_calib.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(display_frame_calib, f"RevDisp: {'ON' if rev_disp_calib else 'OFF'}(r)", (10, display_frame_calib.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        
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
        ret_calib, camera_matrix_calib, dist_coeffs_calib, rvecs_calib, tvecs_calib = \
            cv2.calibrateCamera(obj_points_list, img_points_list, gray_frame_shape, None, None)
        if ret_calib:
            mean_error = 0
            for i in range(len(obj_points_list)):
                imgpoints2, _ = cv2.projectPoints(obj_points_list[i], rvecs_calib[i], tvecs_calib[i], camera_matrix_calib, dist_coeffs_calib)
                error = cv2.norm(img_points_list[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error
            mean_error = mean_error/len(obj_points_list) if len(obj_points_list) > 0 else 0.0
            
            reproj_error_message = f"Mean Reprojection Error: {mean_error:.4f} pixels"
            print(f"Cam Matrix:\n{camera_matrix_calib}\nDist Coeffs:\n{dist_coeffs_calib.ravel()}\n{reproj_error_message}")
            calibration_filepath = os.path.join(DATA_OUTPUT_DIRECTORY, CALIBRATION_FILE)
            try:
                np.savez(calibration_filepath, camera_matrix=camera_matrix_calib, dist_coeffs=dist_coeffs_calib, reprojection_error=mean_error, num_images=len(obj_points_list))
                messagebox.showinfo("Calibration OK", f"Saved: '{calibration_filepath}'.\n{reproj_error_message}", parent=parent_tk_window)
                return camera_matrix_calib, dist_coeffs_calib
            except Exception as e: messagebox.showerror("Save Error", f"Failed to save calib: {e}", parent=parent_tk_window)
        else: messagebox.showwarning("Calib Fail", "cv2.calibrateCamera false.", parent=parent_tk_window)
    else: messagebox.showwarning("Calib Fail", f"Need >3 images (got {len(obj_points_list)}).", parent=parent_tk_window)
    return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

def generate_aruco_marker(marker_id, size_px=600, border_px=50, save=True, parent_tk_window=None):
    # ... (no changes)
    aruco_dict_gen = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    try: 
        img_gray = aruco.generateImageMarker(aruco_dict_gen, marker_id, size_px)
    except cv2.error as e: 
        messagebox.showerror("Marker Gen Err", f"ID {marker_id}: {e}", parent=parent_tk_window); return None
    
    img_bgr_bord = cv2.cvtColor(np.pad(img_gray, border_px, 'constant', constant_values=255), cv2.COLOR_GRAY2BGR)
    font, txt_col, tot_sz = cv2.FONT_HERSHEY_SIMPLEX, (0,0,0), size_px + 2*border_px
    id_txt = f"ID: {marker_id}"; (tw,th),_ = cv2.getTextSize(id_txt,font,0.7,2)
    cv2.putText(img_bgr_bord,id_txt,(tot_sz-tw-max(5,int(border_px*0.1)),tot_sz-max(5,int(border_px*0.1))),font,0.7,txt_col,2)

    if border_px >= 20:
        ax_th, arr_l, tip_l = 2, max(10,int(border_px*0.6)), 0.3
        cx,cy,cz = (0,0,255),(0,255,0),(255,0,0) 
        sfs,sft,pad = 0.5,1,max(3,int(border_px*0.1))
        orig_x = (border_px, border_px + size_px + border_px//2)
        cv2.arrowedLine(img_bgr_bord, orig_x, (orig_x[0]+arr_l, orig_x[1]), cx, ax_th, tipLength=tip_l)
        (xlw,xlh),_ = cv2.getTextSize("X",font,sfs,sft); cv2.putText(img_bgr_bord,"X",(orig_x[0]+arr_l+pad, orig_x[1]+xlh//2),font,sfs,cx,sft)
        orig_y = (border_px - border_px//2, border_px + size_px)
        cv2.arrowedLine(img_bgr_bord, orig_y, (orig_y[0], orig_y[1]-arr_l), cy, ax_th, tipLength=tip_l)
        (ylw,ylh),_ = cv2.getTextSize("Y",font,sfs,sft); cv2.putText(img_bgr_bord,"Y",(orig_y[0]-ylw//2, orig_y[1]-arr_l-pad),font,sfs,cy,sft)
        z_cen,z_rad = (border_px//2,border_px//2),max(5,border_px//4)
        cv2.circle(img_bgr_bord,z_cen,z_rad,cz,ax_th); cv2.circle(img_bgr_bord,z_cen,max(1,z_rad//3),cz,-1)
        (zlw,zlh),_ = cv2.getTextSize("Z",font,sfs,sft); cv2.putText(img_bgr_bord,"Z",(z_cen[0]+z_rad+pad,z_cen[1]+zlh//2),font,sfs,cz,sft)
        (zoutw,zouth),_ = cv2.getTextSize("(out)",font,sfs*0.8,sft); cv2.putText(img_bgr_bord,"(out)",(z_cen[0]+z_rad+pad,z_cen[1]+zlh//2+zouth+2),font,sfs*0.8,cz,sft)

    if marker_id == REFERENCE_MARKER_ID and USE_REFERENCE_MARKER:
        ref_txt = "REFERENCE MARKER"; (trw,trh),_ = cv2.getTextSize(ref_txt,font,0.8,2)
        cv2.putText(img_bgr_bord,ref_txt,((tot_sz-trw)//2,border_px//2+trh//2 if border_px>trh else trh+5),font,0.8,txt_col,2)
    if save:
        fn_sfx = f"_ID{marker_id}" + ('_REFERENCE' if marker_id==REFERENCE_MARKER_ID and USE_REFERENCE_MARKER else '')
        fn = os.path.join(DATA_OUTPUT_DIRECTORY,f"aruco_marker{fn_sfx}.png")
        try: cv2.imwrite(fn,img_bgr_bord); msg = f"Saved: {fn}"
        except Exception as e: msg = f"Save Error: {e}"
        if parent_tk_window: messagebox.showinfo("Marker Gen", msg, parent=parent_tk_window)
        else: print(msg)
    return img_bgr_bord

def create_chessboard(squares_x, squares_y, square_size_px=100, save=True, parent_tk_window=None):
    # ... (no changes)
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
        fn = os.path.join(DATA_OUTPUT_DIRECTORY,f"chessboard_{squares_x}x{squares_y}_corners_{CHESSBOARD_SQUARE_SIZE_MM}mm.png")
        try: cv2.imwrite(fn,final_img); msg = f"Saved: {fn}"
        except Exception as e: msg = f"Save Error: {e}"
        if parent_tk_window: messagebox.showinfo("Chess Gen", msg, parent=parent_tk_window)
        else: print(msg)
    return final_img

def calibrate_buoy_marker_geometry(buoy_id_to_calibrate, parent_tk_window=None, num_frames_to_capture=30):
    # ... (no changes)
    global ALL_BUOY_GEOMETRY_CALIBRATIONS 

    if buoy_id_to_calibrate not in BUOY_ID_TO_MARKER_IDS_MAP:
        messagebox.showerror("Error", f"Buoy ID {buoy_id_to_calibrate} not defined.", parent=parent_tk_window); return False

    target_marker_ids = BUOY_ID_TO_MARKER_IDS_MAP[buoy_id_to_calibrate]
    if len(target_marker_ids) < 2: 
        messagebox.showinfo("Info", f"Buoy ID {buoy_id_to_calibrate} has < 2 markers. No relative calibration needed.", parent=parent_tk_window); return False

    id_X0 = target_marker_ids[0] 
    ids_to_calibrate_relative_to_X0 = target_marker_ids[1:]

    msg = (f"Starting calibration for Buoy ID: {buoy_id_to_calibrate}.\n"
           f"Reference (X0): ID {id_X0}\n"
           f"Markers to calibrate relative to X0: {ids_to_calibrate_relative_to_X0}\n\n"
           f"Ensure the reference marker X0 (ID {id_X0}) AND AT LEAST ONE other target marker "
           f"(from {ids_to_calibrate_relative_to_X0}) are visible together.\n"
           f"Move the buoy slowly, showing X0 with each other marker in turn (aim for at least "
           f"{BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR} views for each X0-Xi pair).\n"
           f"Press 'c' to capture (collecting up to {num_frames_to_capture} good 'c' presses overall).\n"
           f"Press 'q' to finish capture early.")
    messagebox.showinfo("Buoy Calibration", msg, parent=parent_tk_window)

    cam_matrix, dist_coeffs = load_camera_calibration()
    aruco_dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters_obj = aruco.DetectorParameters()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == 'nt' else None)
    if not cap.isOpened(): messagebox.showerror("Cam Error", f"No cam {CAMERA_INDEX}.", parent=parent_tk_window); return False
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
                    cv2.drawFrameAxes(display_frame, cam_matrix, dist_coeffs, rvecs_est[i], tvecs_est[i], MARKER_SIZE_METERS * 0.5)
        
        x0_is_visible = id_X0 in poses_cam_this_frame
        any_xi_visible_with_x0 = False
        if x0_is_visible:
            for marker_id_xi_check in ids_to_calibrate_relative_to_X0:
                if marker_id_xi_check in poses_cam_this_frame:
                    any_xi_visible_with_x0 = True
                    break
        
        status_text_calib, status_color_calib = "Aim: Show X0 + another buoy marker", (0, 165, 255) 
        if x0_is_visible and any_xi_visible_with_x0:
            status_text_calib = f"OK: X0 & another marker visible. Press 'c'."
            status_color_calib = (0, 255, 0) 
        elif x0_is_visible and not any_xi_visible_with_x0:
            status_text_calib = f"X0 (ID {id_X0}) visible. Show another (from {ids_to_calibrate_relative_to_X0}) with it."
            status_color_calib = (0, 255, 255) 
        elif not x0_is_visible:
            status_text_calib = f"X0 (ID {id_X0}) NOT visible. Make it visible."
            status_color_calib = (0, 0, 255) 
        
        cv2.putText(display_frame, status_text_calib, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color_calib, 2)
        cv2.putText(display_frame, f"Buoy ID: {buoy_id_to_calibrate} (X0: {id_X0})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Good 'c' Presses: {frames_captured_count}/{num_frames_to_capture}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset_counts = 120
        for xi_id_disp in ids_to_calibrate_relative_to_X0:
            count_xi = len(collected_relative_poses[xi_id_disp]['tvecs'])
            cv2.putText(display_frame, f"ID {xi_id_disp} (vs X0) samples: {count_xi}", (10, y_offset_counts), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 50), 1)
            y_offset_counts += 20

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): break
        elif key == ord('c'):
            if id_X0 not in poses_cam_this_frame:
                print(f"Capture Press Skipped: Reference marker X0 (ID {id_X0}) not visible.")
                if parent_tk_window: 
                    messagebox.showwarning("Capture Skipped", f"Reference marker X0 (ID {id_X0}) not visible. Cannot capture relative poses.", parent=parent_tk_window)
                continue 

            rvec_X0_cam, tvec_X0_cam = poses_cam_this_frame[id_X0]
            captured_at_least_one_pair_this_c_press = False
            
            for marker_id_Xi in ids_to_calibrate_relative_to_X0:
                if marker_id_Xi in poses_cam_this_frame: 
                    try:
                        rvec_Xi_cam, tvec_Xi_cam = poses_cam_this_frame[marker_id_Xi]
                        rvec_X0_Xi, tvec_X0_Xi = transform_pose_to_reference(rvec_Xi_cam, tvec_Xi_cam, rvec_X0_cam, tvec_X0_cam)
                        
                        collected_relative_poses[marker_id_Xi]['tvecs'].append(tvec_X0_Xi.flatten())
                        collected_relative_poses[marker_id_Xi]['rvecs'].append(rvec_X0_Xi.flatten())
                        print(f"  Captured relative pose for X0 (ID {id_X0}) <-- Xi (ID {marker_id_Xi})")
                        captured_at_least_one_pair_this_c_press = True
                    except Exception as e:
                        print(f"Error calculating/storing relative pose for X0 (ID {id_X0}) and Xi (ID {marker_id_Xi}): {e}")
            
            if captured_at_least_one_pair_this_c_press:
                frames_captured_count += 1
                print(f"Successful data capture from this 'c' press. Total good 'c' presses: {frames_captured_count}/{num_frames_to_capture}")
            else:
                msg_info = f"X0 (ID {id_X0}) was visible, but no other target buoy markers (from {ids_to_calibrate_relative_to_X0}) were simultaneously visible to form a pair."
                print(f"Capture Press Info: {msg_info}")
                if parent_tk_window:
                    messagebox.showinfo("Capture Info", msg_info, parent=parent_tk_window)
    
    cap.release(); cv2.destroyAllWindows()

    found_any_data_at_all = any(len(data['tvecs']) > 0 for data in collected_relative_poses.values())

    if not found_any_data_at_all and frames_captured_count == 0 : 
        messagebox.showwarning("Calibration Incomplete", "No valid data pairs (X0, Xi) were captured at all across all 'c' presses.", parent=parent_tk_window)
        return False
    
    problematic_markers_low_samples = []
    perfectly_missed_markers = []

    for marker_id_Xi_check, data_check in collected_relative_poses.items():
        num_samples = len(data_check['tvecs'])
        if num_samples == 0:
            perfectly_missed_markers.append(marker_id_Xi_check)
            print(f"Warning: No samples collected for marker ID {marker_id_Xi_check} relative to X0.")
        elif num_samples < BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR:
            problematic_markers_low_samples.append(marker_id_Xi_check)
            print(f"Warning: Only {num_samples} samples for marker ID {marker_id_Xi_check} (less than min {BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR}).")
        else: 
            print(f"Collected {num_samples} samples for marker ID {marker_id_Xi_check} relative to X0 (min required: {BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR}).")

    warning_messages_list = []
    if perfectly_missed_markers:
        warning_messages_list.append(f"Markers with NO samples: {', '.join(map(str, perfectly_missed_markers))}. Their poses will be stored as 'None'.")
    if problematic_markers_low_samples:
        warning_messages_list.append(f"Markers with FEW samples (<{BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR}): {', '.join(map(str, problematic_markers_low_samples))}. Their calibration may be less accurate.")

    if warning_messages_list:
        full_warning_msg_str = "\n".join(warning_messages_list) + "\n\nProceed with averaging and saving?"
        if not messagebox.askyesno("Potential Calibration Issues", full_warning_msg_str, icon='warning', parent=parent_tk_window):
            print("Calibration aborted by user due to sample count issues.")
            return False 

    print(f"\nCalculating avg poses for Buoy ID {buoy_id_to_calibrate} from collected X0-Xi pairs...")
    buoy_calibration_results_for_json = {} 
    for marker_id_Xi_process, data_collected in collected_relative_poses.items():
        if not data_collected['tvecs'] or not data_collected['rvecs']: 
            print(f"  Marker {marker_id_Xi_process}: No data collected. Storing as None."); 
            buoy_calibration_results_for_json[str(marker_id_Xi_process)] = {'t_rel_to_X0': None, 'rvec_rel_to_X0': None}
            continue
        
        avg_tvec_X0_Xi = np.mean(np.array(data_collected['tvecs']), axis=0)
        
        quats_X0_Xi = []
        for r_val_X0_Xi in data_collected['rvecs']:
            if r_val_X0_Xi is not None:
                try:
                    R_mat_X0_Xi, _ = cv2.Rodrigues(np.array(r_val_X0_Xi, dtype=np.float32))
                    quats_X0_Xi.append(R.from_matrix(R_mat_X0_Xi).as_quat())
                except (cv2.error, ValueError): pass 

        if quats_X0_Xi:
            avg_quat_X0_Xi = average_quaternions_weighted(quats_X0_Xi, [1.0]*len(quats_X0_Xi)) 
            avg_rvec_X0_Xi = R.from_quat(avg_quat_X0_Xi).as_rotvec()
        else:
            avg_rvec_X0_Xi = np.array([0.,0.,0.], dtype=np.float32) # Fallback

        print(f"  Marker {marker_id_Xi_process} relative to X0 (ID {id_X0}): tvec={avg_tvec_X0_Xi}, rvec={avg_rvec_X0_Xi}")
        buoy_calibration_results_for_json[str(marker_id_Xi_process)] = {
            't_rel_to_X0': avg_tvec_X0_Xi.astype(np.float32), 
            'rvec_rel_to_X0': avg_rvec_X0_Xi.astype(np.float32)
        }

    ALL_BUOY_GEOMETRY_CALIBRATIONS[str(buoy_id_to_calibrate)] = buoy_calibration_results_for_json
    save_buoy_geometry_calibrations(ALL_BUOY_GEOMETRY_CALIBRATIONS)
    initialize_buoy_configs() 

    messagebox.showinfo("Calibration Complete", f"Calibration for Buoy ID {buoy_id_to_calibrate} finished.\nResults saved. Runtime config updated.", parent=parent_tk_window)
    return True

#################################
# GUI CLASSES
#################################
class OfflineProcessingGUI(tk.Toplevel):
    # ... (no changes)
    def __init__(self, master):
        super().__init__(master)
        self.title("Offline Video Processor")
        self.geometry("500x380") 
        self.transient(master); self.grab_set()

        ttk.Label(self, text="Select video for offline ArUco processing.").pack(pady=(10,5))
        file_frame = ttk.Frame(self); file_frame.pack(pady=5, padx=10, fill=tk.X)
        self.filepath_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.filepath_var, width=40, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT)

        self.ref_marker_var = tk.BooleanVar(value=USE_REFERENCE_MARKER)
        self.ref_marker_check = ttk.Checkbutton(self, text=f"Use Ref Marker (ID: {REFERENCE_MARKER_ID}) for this run", variable=self.ref_marker_var)
        self.ref_marker_check.pack(pady=5)

        time_input_frame = ttk.LabelFrame(self, text="Optional: Video Start Timestamp", padding="5")
        time_input_frame.pack(pady=5, padx=10, fill=tk.X)

        self.video_start_date_var = tk.StringVar()
        self.video_start_time_var = tk.StringVar()

        ttk.Label(time_input_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(time_input_frame, textvariable=self.video_start_date_var, width=15).grid(row=0, column=1, sticky=tk.EW, padx=2, pady=2)

        ttk.Label(time_input_frame, text="Start Time (HH:MM:SS):").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(time_input_frame, textvariable=self.video_start_time_var, width=15).grid(row=1, column=1, sticky=tk.EW, padx=2, pady=2)
        time_input_frame.columnconfigure(1, weight=1) 

        self.process_button = ttk.Button(self, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.process_button.pack(pady=10)
        self.status_label = ttk.Label(self, text="Status: Idle"); self.status_label.pack(pady=5, fill=tk.X, padx=10)
        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=100, mode='determinate'); self.progress_bar.pack(pady=(5,10), fill=tk.X, padx=10)

    def browse_file(self):
        filename = filedialog.askopenfilename(title="Select Video", filetypes=(("Video files", "*.mov *.mp4 *.avi *.mkv"),("All files", "*.*")), parent=self)
        if filename:
            self.filepath_var.set(filename); self.process_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Selected: {os.path.basename(filename)}"); self.progress_bar["value"] = 0

    def start_processing(self):
        video_path = self.filepath_var.get()
        if not video_path: messagebox.showerror("Error", "No video selected.", parent=self); return
        use_ref_for_this_run = self.ref_marker_var.get()

        video_start_date = self.video_start_date_var.get()
        video_start_time = self.video_start_time_var.get()

        self.process_button.config(state=tk.DISABLED); self.browse_button.config(state=tk.DISABLED); self.ref_marker_check.config(state=tk.DISABLED)
        for child in self.winfo_children(): 
            if isinstance(child, ttk.LabelFrame):
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Entry):
                        widget.config(state=tk.DISABLED)

        self.status_label.config(text="Processing..."); self.progress_bar["value"] = 0; self.update_idletasks()
        try:
            process_video_offline(video_path, use_ref_for_this_run,
                                  self.status_label, self.progress_bar,
                                  video_start_date, video_start_time)
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error: {e}", parent=self); self.status_label.config(text=f"Error: {e}")
        finally:
            self.process_button.config(state=tk.NORMAL if self.filepath_var.get() else tk.DISABLED)
            self.browse_button.config(state=tk.NORMAL); self.ref_marker_check.config(state=tk.NORMAL)
            for child in self.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for widget in child.winfo_children():
                        if isinstance(widget, ttk.Entry):
                           widget.config(state=tk.NORMAL)

class SettingsDialog(tk.Toplevel):
    # ... (no changes from previous KF version, KF params are general enough)
    def __init__(self, master):
        super().__init__(master)
        self.title("Application Settings"); self.geometry("500x600") 
        self.transient(master); self.grab_set()
        self.initial_ref_marker_id = REFERENCE_MARKER_ID

        self.use_ref_marker_var = tk.BooleanVar(value=USE_REFERENCE_MARKER)
        self.ref_marker_id_var = tk.IntVar(value=REFERENCE_MARKER_ID)
        self.record_video_var = tk.BooleanVar(value=RECORD_VIDEO)
        self.reverse_display_var = tk.BooleanVar(value=REVERSE_CAMERA_DISPLAY)
        self.log_all_marker_data_var = tk.BooleanVar(value=LOG_ALL_MARKER_DATA) 
        self.camera_index_var = tk.IntVar(value=CAMERA_INDEX)
        self.marker_size_var = tk.DoubleVar(value=MARKER_SIZE_METERS)
        self.cam_res_width_var = tk.IntVar(value=CAMERA_RESOLUTION[0])
        self.cam_res_height_var = tk.IntVar(value=CAMERA_RESOLUTION[1])

        self.use_kalman_filter_var = tk.BooleanVar(value=USE_KALMAN_FILTER)
        self.kf_proc_noise_vel_pos_var = tk.DoubleVar(value=KF_PROCESS_NOISE_VEL_POS)
        self.kf_proc_noise_vel_ori_var = tk.DoubleVar(value=KF_PROCESS_NOISE_VEL_ORI)
        self.kf_meas_noise_pos_var = tk.DoubleVar(value=KF_MEASUREMENT_NOISE_POS)
        self.kf_meas_noise_ori_var = tk.DoubleVar(value=KF_MEASUREMENT_NOISE_ORI)

        main_frame = ttk.Frame(self, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        r=0
        ttk.Label(main_frame, text="Ref Marker:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(main_frame, variable=self.use_ref_marker_var, text="Enable", command=self._toggle_ref_id_entry).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        ttk.Label(main_frame, text="Ref Marker ID:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.ref_id_entry = ttk.Entry(main_frame, textvariable=self.ref_marker_id_var, width=7); self.ref_id_entry.grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        
        ttk.Label(main_frame, text="Default Record Video (Live):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(main_frame, variable=self.record_video_var, text="On").grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        ttk.Label(main_frame, text="Default Reverse Display:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(main_frame, variable=self.reverse_display_var, text="On").grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
                
        ttk.Label(main_frame, text="Log All Marker Data:").grid(row=r, column=0, sticky=tk.W, pady=2) 
        ttk.Checkbutton(main_frame, variable=self.log_all_marker_data_var, text="Enable (writes separate detailed marker CSV)").grid(row=r, column=1, columnspan=2, sticky=tk.W, pady=2); r+=1

        ttk.Label(main_frame, text="Camera Index:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main_frame, textvariable=self.camera_index_var, width=7).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        ttk.Label(main_frame, text="ArUco Marker Size (m):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main_frame, textvariable=self.marker_size_var, width=7).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        
        ttk.Label(main_frame, text="Cam Res Width:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main_frame, textvariable=self.cam_res_width_var, width=7).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        ttk.Label(main_frame, text="Cam Res Height:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main_frame, textvariable=self.cam_res_height_var, width=7).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=r, column=0, columnspan=3, sticky="ew", pady=10); r+=1
        kf_label = ttk.Label(main_frame, text="Kalman Filter (Buoy CoG):", font=('Helvetica', 10, 'bold'))
        kf_label.grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=(5,2)); r+=1
        
        ttk.Label(main_frame, text="Enable Kalman Filter:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(main_frame, variable=self.use_kalman_filter_var, text="On").grid(row=r, column=1, sticky=tk.W, pady=2); r+=1

        ttk.Label(main_frame, text="Proc. Noise Vel Pos (m/s):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main_frame, textvariable=self.kf_proc_noise_vel_pos_var, width=10).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        ttk.Label(main_frame, text="Proc. Noise Vel Ori (rad/s):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main_frame, textvariable=self.kf_proc_noise_vel_ori_var, width=10).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        ttk.Label(main_frame, text="Meas. Noise Pos (m):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main_frame, textvariable=self.kf_meas_noise_pos_var, width=10).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        ttk.Label(main_frame, text="Meas. Noise Ori (rad):").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main_frame, textvariable=self.kf_meas_noise_ori_var, width=10).grid(row=r, column=1, sticky=tk.W, pady=2); r+=1
        
        btn_fr = ttk.Frame(main_frame); btn_fr.grid(row=r, column=0, columnspan=3, pady=(15,5)) 
        ttk.Button(btn_fr, text="Apply", command=self.apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_fr, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        self._toggle_ref_id_entry()

    def _toggle_ref_id_entry(self): self.ref_id_entry.config(state=tk.NORMAL if self.use_ref_marker_var.get() else tk.DISABLED)
    
    def apply_settings(self):
        global USE_REFERENCE_MARKER,REFERENCE_MARKER_ID,RECORD_VIDEO,REVERSE_CAMERA_DISPLAY 
        global CAMERA_INDEX, MARKER_SIZE_METERS, CAMERA_RESOLUTION, LOG_ALL_MARKER_DATA
        global USE_KALMAN_FILTER, KF_PROCESS_NOISE_VEL_POS, KF_PROCESS_NOISE_VEL_ORI
        global KF_MEASUREMENT_NOISE_POS, KF_MEASUREMENT_NOISE_ORI

        n_use_ref = self.use_ref_marker_var.get()
        n_rec = self.record_video_var.get()
        n_rev = self.reverse_display_var.get()
        n_log_all_markers = self.log_all_marker_data_var.get()
        n_use_kf = self.use_kalman_filter_var.get()

        v_ref_id = self.initial_ref_marker_id
        if n_use_ref:
            try:
                tid = self.ref_marker_id_var.get()
                if tid >= 0: v_ref_id = tid
                else: messagebox.showwarning("Inv ID","Ref ID non-neg.",parent=self); self.ref_marker_id_var.set(self.initial_ref_marker_id); return
            except tk.TclError: messagebox.showwarning("Inv ID","Ref ID int.",parent=self); self.ref_marker_id_var.set(self.initial_ref_marker_id); return
        
        try:
            n_cam_idx = self.camera_index_var.get()
            if n_cam_idx < 0: raise ValueError("Cam index must be >= 0.")
            n_marker_size = self.marker_size_var.get()
            if n_marker_size <=0: raise ValueError("Marker size must be > 0.")
            n_cam_w = self.cam_res_width_var.get()
            n_cam_h = self.cam_res_height_var.get()
            if n_cam_w <=0 or n_cam_h <=0: raise ValueError("Cam res must be > 0.")

            n_kf_pnvp = self.kf_proc_noise_vel_pos_var.get()
            n_kf_pnvo = self.kf_proc_noise_vel_ori_var.get()
            n_kf_mnp = self.kf_meas_noise_pos_var.get()
            n_kf_mno = self.kf_meas_noise_ori_var.get()
            if any(v < 0 for v in [n_kf_pnvp, n_kf_pnvo, n_kf_mnp, n_kf_mno]):
                raise ValueError("Kalman filter noise sigmas must be non-negative.")

        except (tk.TclError, ValueError) as e:
             messagebox.showwarning("Invalid Input", f"Error in settings: {e}", parent=self); return

        USE_REFERENCE_MARKER,REFERENCE_MARKER_ID = n_use_ref, v_ref_id
        RECORD_VIDEO,REVERSE_CAMERA_DISPLAY = n_rec,n_rev
        LOG_ALL_MARKER_DATA = n_log_all_markers
        CAMERA_INDEX, MARKER_SIZE_METERS = n_cam_idx, n_marker_size
        CAMERA_RESOLUTION = (n_cam_w, n_cam_h)

        USE_KALMAN_FILTER = n_use_kf
        KF_PROCESS_NOISE_VEL_POS = n_kf_pnvp
        KF_PROCESS_NOISE_VEL_ORI = n_kf_pnvo
        KF_MEASUREMENT_NOISE_POS = n_kf_mnp
        KF_MEASUREMENT_NOISE_ORI = n_kf_mno
        
        print(f"Settings updated: Ref={USE_REFERENCE_MARKER},ID={REFERENCE_MARKER_ID},Rec={RECORD_VIDEO},Rev={REVERSE_CAMERA_DISPLAY},LogAllMarkers={LOG_ALL_MARKER_DATA},CamIdx={CAMERA_INDEX},MarkerSize={MARKER_SIZE_METERS},CamRes={CAMERA_RESOLUTION}")
        print(f"KF Settings: UseKF={USE_KALMAN_FILTER}, ProcNoiseVelPos={KF_PROCESS_NOISE_VEL_POS}, ProcNoiseVelOri={KF_PROCESS_NOISE_VEL_ORI}, MeasNoisePos={KF_MEASUREMENT_NOISE_POS}, MeasNoiseOri={KF_MEASUREMENT_NOISE_ORI}")
        initialize_buoy_configs() 
        messagebox.showinfo("Settings OK","Global settings updated. Buoy configs re-initialized.",parent=self); self.destroy()

class BuoyTrackerApp:
    # ... (no changes)
    def __init__(self, master):
        self.master = master; master.title("ArUco Buoy Tracker & Tools"); master.geometry("450x550")
        mf = ttk.Frame(master,padding="10"); mf.pack(fill=tk.BOTH,expand=True); ttk.Style().configure("TLabelframe.Label",font=("Helvetica",10,"bold"))
        
        gf=ttk.LabelFrame(mf,text="Generation Tools",padding="10");gf.pack(fill=tk.X,pady=(10,5),padx=5)
        ttk.Button(gf,text="Generate Single ArUco Marker",command=self.ui_generate_marker).pack(fill=tk.X,pady=(4,2))
        ttk.Button(gf,text="Generate Reference Marker",command=self.ui_generate_reference_marker).pack(fill=tk.X,pady=(4,2))
        ttk.Button(gf,text="Generate Chessboard Pattern",command=self.ui_generate_chessboard).pack(fill=tk.X,pady=(4,2))

        pf=ttk.LabelFrame(mf,text="Processing & Calibration Tools",padding="10");pf.pack(fill=tk.X,pady=(10,5),padx=5)
        ttk.Button(pf,text="Calibrate Camera (Intrinsic)",command=self.ui_calibrate_camera).pack(fill=tk.X,pady=(4,2))
        ttk.Button(pf,text="Calibrate Buoy Marker Geometry",command=self.ui_calibrate_buoy_geometry).pack(fill=tk.X,pady=(4,2))
        ttk.Button(pf,text="Start Live Tracking",command=self.ui_live_tracking).pack(fill=tk.X,pady=(4,2))
        ttk.Button(pf,text="Launch Offline Video Processor",command=self.ui_launch_offline_processor).pack(fill=tk.X,pady=(4,2))
        
        af=ttk.LabelFrame(mf,text="Application",padding="10");af.pack(fill=tk.X,pady=(10,5),padx=5)
        ttk.Button(af,text="Configure Settings",command=self.ui_configure_settings).pack(fill=tk.X,pady=(4,2))
        ttk.Button(af,text="Exit",command=master.quit).pack(fill=tk.X,pady=(4,2))

        self.status_var=tk.StringVar(value="Ready.");ttk.Label(master,textvariable=self.status_var,relief=tk.SUNKEN,anchor=tk.W,padding=2).pack(side=tk.BOTTOM,fill=tk.X)

    def _gi(self,t,p,iv=None,mn=None,mx=None): return simpledialog.askinteger(t,p,parent=self.master,initialvalue=iv,minvalue=mn,maxvalue=mx)
    
    def ui_generate_marker(self):
        dict_size = 250 
        if ARUCO_DICT_TYPE == aruco.DICT_4X4_50: dict_size = 50
        elif ARUCO_DICT_TYPE == aruco.DICT_4X4_100: dict_size = 100
        
        mid = self._gi("GenMark",f"ID (0-{dict_size-1}):",0,0,dict_size-1)
        if mid is None: self.status_var.set("Marker gen cancelled."); return
        spx = self._gi("GenMark","Size (px):",600,50,2000)
        if spx is None: self.status_var.set("Marker gen cancelled."); return
        bpx = self._gi("GenMark","Border (px):",30,0,500)
        if bpx is None: self.status_var.set("Marker gen cancelled."); return

        self.status_var.set(f"Gen marker ID {mid}..."); self.master.update_idletasks()
        img=generate_aruco_marker(mid,spx,bpx,True,self.master)
        if img is not None: 
            cv2.imshow(f"Generated ArUco Marker - ID:{mid}",img)
            cv2.waitKey(0); cv2.destroyWindow(f"Generated ArUco Marker - ID:{mid}")
        self.status_var.set("Ready.")

    def ui_generate_reference_marker(self):
        if not USE_REFERENCE_MARKER: messagebox.showinfo("Info", "Ref marker disabled. Gen as regular.", parent=self.master)
        ref_id_gen = REFERENCE_MARKER_ID
        spx = self._gi("GenRefMark",f"Size Ref ID {ref_id_gen}(px):",300,50,2000)
        if spx is None: self.status_var.set("Ref marker gen cancelled."); return
        bpx = self._gi("GenRefMark","Border (px):",30,0,500)
        if bpx is None: self.status_var.set("Ref marker gen cancelled."); return

        self.status_var.set(f"Gen ref marker (ID:{ref_id_gen})..."); self.master.update_idletasks()
        img=generate_aruco_marker(ref_id_gen,spx,bpx,True,self.master)
        if img is not None: 
            cv2.imshow(f"Generated Reference Marker - ID:{ref_id_gen}",img)
            cv2.waitKey(0); cv2.destroyWindow(f"Generated Reference Marker - ID:{ref_id_gen}")
        self.status_var.set("Ready.")

    def ui_generate_chessboard(self):
        sx,sy=CALIBRATION_CHESSBOARD_SHAPE; spx=self._gi("GenChess","Square size (px):",50,20,500)
        if spx is None: self.status_var.set("Chess gen cancelled."); return
        self.status_var.set("Gen chessboard..."); self.master.update_idletasks()
        img=create_chessboard(sx,sy,spx,True,self.master)
        if img is not None: 
            cv2.imshow(f"Generated Chessboard ({sx}x{sy})",img)
            cv2.waitKey(0); cv2.destroyWindow(f"Generated Chessboard ({sx}x{sy})")
        self.status_var.set("Ready.")

    def ui_calibrate_camera(self):
        self.status_var.set("Starting cam calib..."); self.master.update_idletasks()
        messagebox.showinfo("Cam Calib","Follow console. 'c' capture, 'r' reverse, 'q' finish (OpenCV).",parent=self.master)
        calibrate_camera(self.master); self.status_var.set("Ready.")

    def ui_live_tracking(self):
        self.status_var.set("Prep live track..."); self.master.update_idletasks()
        log_all_text = "On (separate detailed marker CSV)" if LOG_ALL_MARKER_DATA else "Off (CoG only CSV)"
        conf_msg = (f"Start Live Tracking?\n\n"
                    f"Cam: {CAMERA_INDEX} @ {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]}\n"
                    f"Ref Marker (ID{REFERENCE_MARKER_ID}): {'Enabled' if USE_REFERENCE_MARKER else 'Disabled'}\n"
                    f"Record Default: {'On' if RECORD_VIDEO else 'Off'}\n"
                    f"Log All Marker Data: {log_all_text}\n"
                    f"Rev Display Default: {'On' if REVERSE_CAMERA_DISPLAY else 'Off'}\n"
                    f"Marker Size: {MARKER_SIZE_METERS*100:.1f} cm\n"
                    f"Kalman Filter: {'Enabled' if USE_KALMAN_FILTER else 'Disabled'}\n"
                    f"Ensure buoy configs are loaded/calibrated.")
        if messagebox.askyesno("Confirm Live",conf_msg,parent=self.master):
            self.status_var.set("Live track active..."); self.master.update_idletasks()
            try: track_buoy_with_aruco_6dof(self.master)
            except Exception as e: messagebox.showerror("Live Track Err",f"Error: {e}",parent=self.master);self.status_var.set(f"Live track err: {e}")
            self.status_var.set("Ready.") 
        else: self.status_var.set("Live track cancelled.")
    
    def ui_launch_offline_processor(self):
        self.status_var.set("Open offline proc..."); self.master.update_idletasks()
        opg = OfflineProcessingGUI(self.master); opg.protocol("WM_DELETE_WINDOW",lambda:[self.status_var.set("Offline proc closed."),opg.destroy()])
    
    def ui_configure_settings(self):
        self.status_var.set("Open settings..."); self.master.update_idletasks()
        sd = SettingsDialog(self.master)
        self.master.wait_window(sd) 
        self.status_var.set("Settings window closed.")

    def ui_calibrate_buoy_geometry(self):
        buoy_ids_available = list(BUOY_ID_TO_MARKER_IDS_MAP.keys())
        if not buoy_ids_available:
            messagebox.showerror("Error", "No buoys defined in BUOY_ID_TO_MARKER_IDS_MAP.", parent=self.master); return

        buoy_id_str = simpledialog.askstring("Calibrate Buoy Geometry",
                                          f"Enter Buoy ID to calibrate (e.g., {', '.join(map(str, buoy_ids_available[:min(3, len(buoy_ids_available))]))}...):",
                                          parent=self.master)
        if buoy_id_str is None: self.status_var.set("Buoy calib cancelled."); return
        try:
            buoy_id_to_calibrate = int(buoy_id_str)
            if buoy_id_to_calibrate not in buoy_ids_available:
                messagebox.showerror("Error", f"Buoy ID {buoy_id_to_calibrate} not valid.", parent=self.master); self.status_var.set("Invalid buoy ID."); return
        except ValueError: messagebox.showerror("Error", "Invalid Buoy ID format.", parent=self.master); self.status_var.set("Invalid buoy ID format."); return
        
        num_other_markers_on_buoy = len(BUOY_ID_TO_MARKER_IDS_MAP[buoy_id_to_calibrate]) - 1
        num_frames_default_suggestion = BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR * max(1, num_other_markers_on_buoy) * 2 
        num_frames_default_suggestion = max(10, num_frames_default_suggestion) 

        prompt_text_num_frames = (f"Number of 'c' presses to capture data.\n"
                                  f"Aim for at least {BUOY_GEOMETRY_CALIB_MIN_SAMPLES_PER_PAIR} views for EACH X0-Xi pair.\n"
                                  f"Buoy {buoy_id_to_calibrate} has {num_other_markers_on_buoy} other marker(s) besides X0.\n"
                                  f"Suggested total 'c' presses: ~{num_frames_default_suggestion}")

        num_frames = self._gi("Buoy Calibration - Samples", prompt_text_num_frames, num_frames_default_suggestion, 5, 500) 
        if num_frames is None: self.status_var.set("Buoy calib cancelled."); return

        self.status_var.set(f"Starting geometry calib for Buoy ID: {buoy_id_to_calibrate}..."); self.master.update_idletasks()
        success = calibrate_buoy_marker_geometry(buoy_id_to_calibrate, self.master, num_frames_to_capture=num_frames)
        self.status_var.set(f"Buoy {buoy_id_to_calibrate} geometry calib {'finished' if success else 'failed/aborted'}.")

#################################
# MAIN EXECUTION
#################################
if __name__ == "__main__":
    os.makedirs(DATA_OUTPUT_DIRECTORY,exist_ok=True)
    os.makedirs(os.path.join(DATA_OUTPUT_DIRECTORY, EXPERIMENTS_BASE_DIRECTORY_NAME), exist_ok=True)

    print("Application starting...")
    print(f"Using SciPy version: {R.random().as_matrix().shape}") # Basic check that SciPy is available

    root = tk.Tk(); app = BuoyTrackerApp(root); root.mainloop()
    print("App closed.")