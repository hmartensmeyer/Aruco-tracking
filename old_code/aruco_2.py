import cv2
import numpy as np
from cv2 import aruco
import time
import csv
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

#################################
# CONFIGURATION PARAMETERS
#################################

# Camera Setup
CAMERA_INDEX = 0
#CAMERA_RESOLUTION = (3840, 2160)
CAMERA_RESOLUTION = (1920, 1080) # 1080p for better performance
CAMERA_FPS = 30
REVERSE_CAMERA_DISPLAY = True

# Video Recording Settings
RECORD_VIDEO = False
VIDEO_CODEC = 'mp4v' # For .mp4 output. Alternatives for MP4: 'XVID' (sometimes for AVI too), 'avc1', 'X264' (if ffmpeg backend)
VIDEO_FPS = 30
VIDEO_EXTENSION = '.mp4' # Use .mp4 for output videos

# ArUco Marker Configuration
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
MARKER_SIZE_METERS = 0.1577

# Reference Marker Configuration
REFERENCE_MARKER_ID = 0
USE_REFERENCE_MARKER = True

# Camera Calibration Parameters
DEFAULT_CAMERA_MATRIX = np.array([
    [1500, 0, CAMERA_RESOLUTION[0]/2], # Example fx, fy, cx, cy
    [0, 1500, CAMERA_RESOLUTION[1]/2],
    [0, 0, 1]
], dtype=np.float32)
DEFAULT_DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

# Calibration Settings
CALIBRATION_CHESSBOARD_SHAPE = (9, 6)
CHESSBOARD_SQUARE_SIZE_MM = 26.63

# Visualization Settings
TRAJECTORY_LENGTH = 30
TRAJECTORY_COLOR = (0, 0, 255)
MARKER_INFO_COLOR = (0, 255, 0)
TIMESTAMP_COLOR = (0, 0, 255)
REFERENCE_MARKER_COLOR = (255, 0, 255)
RECORD_INDICATOR_COLOR = (0, 0, 255)
OFFLINE_PROGRESS_COLOR = (255, 165, 0)
SHOW_OFFLINE_PROCESSING_PREVIEW = True # Set to False to speed up offline processing (no live preview window)
OFFLINE_PREVIEW_RESIZE_FACTOR = 0.25 # e.g., 0.25 for 1/4th size preview

# File Settings
DATA_OUTPUT_DIRECTORY = "buoy_tracking_data"
CALIBRATION_FILE = "camera_calibration.npz"
VIDEO_DIRECTORY = "recordings"

#################################
# UTILITY FUNCTIONS
#################################

def transform_pose_to_reference(marker_rvec, marker_tvec, ref_rvec, ref_tvec):
    ref_rmat, _ = cv2.Rodrigues(ref_rvec)
    marker_rmat, _ = cv2.Rodrigues(marker_rvec)
    ref_rmat_inv = np.transpose(ref_rmat)
    rel_tvec = np.dot(ref_rmat_inv, marker_tvec - ref_tvec)
    rel_rmat = np.dot(ref_rmat_inv, marker_rmat)
    rel_rvec, _ = cv2.Rodrigues(rel_rmat)
    return rel_rvec, rel_tvec

def rotation_vector_to_euler(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, np.zeros((3, 1))))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix.astype(np.float64))[6]
    return euler_angles

def load_camera_calibration():
    calibration_path = os.path.join(DATA_OUTPUT_DIRECTORY, CALIBRATION_FILE)
    try:
        data = np.load(calibration_path)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print(f"Loaded camera calibration parameters from {calibration_path}")
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print(f"Calibration file not found at {calibration_path}. Using default parameters.")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS
    except Exception as e:
        print(f"Error loading calibration file: {e}. Using default parameters.")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

def get_video_writer(frame_size, filename_with_ext, fps=None):
    video_path_dir = os.path.join(DATA_OUTPUT_DIRECTORY, VIDEO_DIRECTORY)
    if not os.path.exists(video_path_dir):
        os.makedirs(video_path_dir)
    
    video_file_full_path = os.path.join(video_path_dir, filename_with_ext)
    
    # Ensure the codec is a 4-character string
    codec_str = VIDEO_CODEC
    if len(codec_str) != 4:
        print(f"Warning: VIDEO_CODEC '{codec_str}' is not 4 characters. Defaulting to 'mp4v'.")
        codec_str = 'mp4v'
        
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    output_fps = fps if fps is not None else VIDEO_FPS

    writer = cv2.VideoWriter(video_file_full_path, fourcc, output_fps, frame_size)
    
    if not writer.isOpened():
        print(f"Warning: Could not initialize video writer with codec {codec_str} for {filename_with_ext} at {output_fps} FPS.")
        # Try a very common fallback for mp4 if the primary fails (e.g. X264 if primary was mp4v)
        # This part might need more robust handling or user configuration
        if codec_str != 'XVID': # XVID is often for AVI, try it as a general fallback
            print("Trying fallback codec XVID (might result in .avi effectively)...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # This usually makes an AVI
            # If we use XVID, the filename extension should ideally be .avi
            # For simplicity, we'll stick to the requested .mp4 but it might not play well.
            # A better approach is to adjust filename_with_ext if codec changes extension preference.
            # For now, let's assume the user configures VIDEO_CODEC correctly for VIDEO_EXTENSION
            writer = cv2.VideoWriter(video_file_full_path, fourcc, output_fps, frame_size)
        
        if not writer.isOpened():
            print(f"Error: Failed to create video writer for {video_file_full_path}. Video recording/annotation disabled.")
            return None
    
    print(f"Video will be saved to: {video_file_full_path}")
    return writer

#################################
# CORE PROCESSING FUNCTION
#################################

def _process_frame_common(
    gray_frame, 
    color_frame_to_annotate, 
    camera_matrix, 
    dist_coeffs, 
    aruco_dict, 
    aruco_parameters, 
    marker_size_m,
    use_reference_marker_flag,
    reference_marker_id_val,
    positions_history,
    rotations_history, # Currently not used for drawing trajectory line, but populated
    trajectory_length_config,
    log_date_str,
    log_time_str
    ):
    log_entries = []
    annotated_frame = color_frame_to_annotate.copy()

    corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, parameters=aruco_parameters)

    ref_marker_detected_this_frame = False
    ref_rvec, ref_tvec = None, None

    if ids is not None and len(ids) > 0:
        # Subpixel refinement of corners
        refined_corners_list = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for corner_set in corners:
            # Ensure corner_set is float32 for cornerSubPix
            refined_corner = cv2.cornerSubPix(gray_frame, corner_set.astype(np.float32), (5, 5), (-1, -1), criteria)
            refined_corners_list.append(refined_corner)
        
        # Draw original detected markers (visually less confusing than subpixel shifts)
        aruco.drawDetectedMarkers(annotated_frame, corners, ids)
        
        # Estimate pose using refined corners
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            refined_corners_list, marker_size_m, camera_matrix, dist_coeffs)

        if use_reference_marker_flag:
            for i, marker_id_arr in enumerate(ids):
                marker_id = marker_id_arr[0]
                if marker_id == reference_marker_id_val:
                    ref_marker_detected_this_frame = True
                    ref_rvec, ref_tvec = rvecs[i][0], tvecs[i][0]
                    cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs,
                                     ref_rvec, ref_tvec, marker_size_m, 3)
                    # Use original corners for text placement
                    c_orig = corners[i][0]
                    center_px_orig = (int(np.mean(c_orig[:, 0])), int(np.mean(c_orig[:, 1])))
                    cv2.putText(annotated_frame, "REFERENCE",
                                (center_px_orig[0] - 40, center_px_orig[1] - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_MARKER_COLOR, 2)
                    break
        
        for i, marker_id_arr in enumerate(ids):
            marker_id = marker_id_arr[0]
            marker_rvec_cam, marker_tvec_cam = rvecs[i][0], tvecs[i][0]

            if use_reference_marker_flag and ref_marker_detected_this_frame and marker_id != reference_marker_id_val:
                log_rvec, log_tvec = transform_pose_to_reference(
                    marker_rvec_cam, marker_tvec_cam, ref_rvec, ref_tvec)
            else:
                log_rvec, log_tvec = marker_rvec_cam, marker_tvec_cam

            cv2.drawFrameAxes(annotated_frame, camera_matrix, dist_coeffs,
                             marker_rvec_cam, marker_tvec_cam, marker_size_m / 2)

            # Use original corners for text placement and trajectory points (pixel coordinates)
            c_orig = corners[i][0]
            center_px_orig = (int(np.mean(c_orig[:, 0])), int(np.mean(c_orig[:, 1])))
            
            euler_angles_cam = rotation_vector_to_euler(marker_rvec_cam)
            euler_angles_log = rotation_vector_to_euler(log_rvec)

            if marker_id not in positions_history:
                positions_history[marker_id] = []
                rotations_history[marker_id] = [] # For potential future use
            
            positions_history[marker_id].append(center_px_orig)
            rotations_history[marker_id].append(euler_angles_log)


            if len(positions_history[marker_id]) > trajectory_length_config:
                positions_history[marker_id].pop(0)
                rotations_history[marker_id].pop(0)

            log_entry = [log_date_str, log_time_str, marker_id]
            if USE_REFERENCE_MARKER: # Global config dictates header structure
                 log_entry.append("1" if ref_marker_detected_this_frame else "0")
            
            log_entry.extend([
                f"{log_tvec[0]:.6f}", f"{log_tvec[1]:.6f}", f"{log_tvec[2]:.6f}",
                f"{euler_angles_log[0][0]:.3f}", f"{euler_angles_log[1][0]:.3f}", f"{euler_angles_log[2][0]:.3f}"
            ])

            if USE_REFERENCE_MARKER:
                log_entry.extend([
                    f"{marker_tvec_cam[0]:.6f}", f"{marker_tvec_cam[1]:.6f}", f"{marker_tvec_cam[2]:.6f}",
                    f"{euler_angles_cam[0][0]:.3f}", f"{euler_angles_cam[1][0]:.3f}", f"{euler_angles_cam[2][0]:.3f}"
                ])
            log_entries.append(log_entry)

            cv2.putText(annotated_frame, f"ID: {marker_id}",
                        (center_px_orig[0] + 10, center_px_orig[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARKER_INFO_COLOR, 2)
            
            pos_text_prefix = "Pos"
            if use_reference_marker_flag and ref_marker_detected_this_frame and marker_id != reference_marker_id_val:
                pos_text_prefix = "Rel Pos"
            
            cv2.putText(annotated_frame, f"{pos_text_prefix}: ({log_tvec[0]:.2f}, {log_tvec[1]:.2f}, {log_tvec[2]:.2f})m",
                        (center_px_orig[0] + 10, center_px_orig[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, MARKER_INFO_COLOR, 1)

            rot_text_prefix = "Rot"
            if use_reference_marker_flag and ref_marker_detected_this_frame and marker_id != reference_marker_id_val:
                rot_text_prefix = "Rel Rot"

            cv2.putText(annotated_frame,
                        f"{rot_text_prefix}: ({euler_angles_log[0][0]:.1f}, {euler_angles_log[1][0]:.1f}, {euler_angles_log[2][0]:.1f})deg",
                        (center_px_orig[0] + 10, center_px_orig[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, MARKER_INFO_COLOR, 1)

    for marker_id_val, pos_list in positions_history.items():
        if len(pos_list) > 1:
            for i in range(1, len(pos_list)):
                cv2.line(annotated_frame, pos_list[i-1], pos_list[i], TRAJECTORY_COLOR, 2)
    
    return annotated_frame, log_entries, positions_history, rotations_history, ref_marker_detected_this_frame

#################################
# MAIN APPLICATION FUNCTIONS
#################################

def track_buoy_with_aruco_6dof():
    # ... (Setup is largely the same)
    timestamp_str_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(DATA_OUTPUT_DIRECTORY, f"buoy_tracking_6dof_{timestamp_str_file}.csv")
    # Video filename construction
    video_filename_base = f"buoy_tracking_live_{timestamp_str_file}{VIDEO_EXTENSION}"
    # ... (rest of the live tracking logic, ensuring video_writer gets filename with .mp4)

    if not os.path.exists(DATA_OUTPUT_DIRECTORY):
        os.makedirs(DATA_OUTPUT_DIRECTORY)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    camera_matrix, dist_coeffs = load_camera_calibration()
    aruco_dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE) # Renamed to avoid conflict
    parameters_obj = aruco.DetectorParameters() # Renamed
    
    positions_live = {} 
    rotations_live = {} 
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['Date', 'Time', 'Marker_ID']
        if USE_REFERENCE_MARKER: # Global config
            header.extend(['Ref_Marker_Detected', 'Pos_X', 'Pos_Y', 'Pos_Z', 
                           'Rot_X', 'Rot_Y', 'Rot_Z',
                           'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z', 
                           'Cam_Rot_X', 'Cam_Rot_Y', 'Cam_Rot_Z'])
        else:
            header.extend(['Pos_X', 'Pos_Y', 'Pos_Z', 'Rot_X', 'Rot_Y', 'Rot_Z'])
        csv_writer.writerow(header)
    
    record_video_flag = RECORD_VIDEO 
    video_writer = None
    # video_filename_base already defined with .mp4

    ret, first_frame = cap.read()
    frame_size = (0,0)
    if ret:
        frame_size = (first_frame.shape[1], first_frame.shape[0])
        if record_video_flag:
            video_writer = get_video_writer(frame_size, video_filename_base, fps=VIDEO_FPS) 
            if video_writer is None: record_video_flag = False
    else:
        print("Error: Failed to get first frame from camera.")
        cap.release()
        return

    print(f"Starting buoy tracking. Data will be saved to {csv_filename}. Press 'q' to quit.")
    # ... (rest of print statements)
    
    reverse_display_flag = REVERSE_CAMERA_DISPLAY
    ref_marker_detected_status = False

    current_video_filename_live = video_filename_base # Store the current recording filename

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        process_frame = frame.copy() 
        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        
        now_ts = datetime.now() # Renamed
        date_str = now_ts.strftime("%Y-%m-%d")
        time_str = now_ts.strftime("%H:%M:%S.%f")[:-3]

        annotated_frame, log_entries, positions_live, rotations_live, ref_marker_detected_status = _process_frame_common(
            gray, process_frame, camera_matrix, dist_coeffs, aruco_dict_obj, parameters_obj,
            MARKER_SIZE_METERS, USE_REFERENCE_MARKER, REFERENCE_MARKER_ID,
            positions_live, rotations_live, TRAJECTORY_LENGTH, date_str, time_str
        )

        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for entry in log_entries:
                csv_writer.writerow(entry)

        # ... (Add live-specific status information as before) ...
        status_text = f"Ref marker {REFERENCE_MARKER_ID}: {'Detected' if ref_marker_detected_status else 'Not Detected'}"
        cv2.putText(annotated_frame, status_text,
                    (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    REFERENCE_MARKER_COLOR if ref_marker_detected_status else (0, 0, 255), 2)
        
        cv2.putText(annotated_frame, f"Reversal: {'ON' if reverse_display_flag else 'OFF'} (r)",
                    (10, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        cv2.putText(annotated_frame, f"{date_str} {time_str}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TIMESTAMP_COLOR, 2)
        
        if record_video_flag:
            radius = 10 
            #pulse = int(5 * np.sin(time.time() * 5) + 10) # Removed pulse for simplicity
            cv2.circle(annotated_frame, (annotated_frame.shape[1] - 30, 30), radius, RECORD_INDICATOR_COLOR, -1) 
            cv2.putText(annotated_frame, "REC",
                        (annotated_frame.shape[1] - 75, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RECORD_INDICATOR_COLOR, 2)
            cv2.putText(annotated_frame, f"Recording: ON (v)",
                        (10, annotated_frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RECORD_INDICATOR_COLOR, 2)
        else:
            cv2.putText(annotated_frame, f"Recording: OFF (v)",
                        (10, annotated_frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)


        display_output_frame = annotated_frame.copy()
        if reverse_display_flag:
            display_output_frame = cv2.flip(display_output_frame, 1)
        
        if record_video_flag and video_writer is not None:
            video_writer.write(annotated_frame) 
        
        cv2.imshow('Buoy 6DOF Tracking (Live)', display_output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reverse_display_flag = not reverse_display_flag
        elif key == ord('v'):
            if record_video_flag:
                record_video_flag = False
                if video_writer is not None:
                    video_writer.release()
                    print(f"Video recording stopped. File: {current_video_filename_live}") 
                    video_writer = None
            else:
                record_video_flag = True
                if video_writer is None:
                    current_rec_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_video_filename_live = f"buoy_tracking_live_{current_rec_timestamp}{VIDEO_EXTENSION}"
                    video_writer = get_video_writer(frame_size, current_video_filename_live, fps=VIDEO_FPS)
                    if video_writer is None:
                        record_video_flag = False
                    else:
                        print(f"Video recording started. Saving to {current_video_filename_live}")
    
    if record_video_flag and video_writer is not None:
        video_writer.release()
        print(f"Final video recording saved to {current_video_filename_live}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Tracking data saved to {csv_filename}")


def process_video_offline(input_video_path, use_ref_marker_override, gui_status_label=None):
    if not os.path.exists(input_video_path):
        print(f"Error: Input video file not found at {input_video_path}")
        if gui_status_label: gui_status_label.config(text=f"Error: File not found.")
        return

    if not os.path.exists(DATA_OUTPUT_DIRECTORY): os.makedirs(DATA_OUTPUT_DIRECTORY)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        if gui_status_label: gui_status_label.config(text=f"Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps_original = cap.get(cv2.CAP_PROP_FPS) # Original FPS for timestamp calculation if POS_MSEC fails
    if video_fps_original == 0: video_fps_original = VIDEO_FPS # Fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if gui_status_label: gui_status_label.config(text=f"Processing: {os.path.basename(input_video_path)}...")
    print(f"Processing video: {input_video_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS (from file): {video_fps_original:.2f}, Total Frames: {total_frames}")

    camera_matrix, dist_coeffs = load_camera_calibration()
    aruco_dict_obj = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters_obj = aruco.DetectorParameters()

    positions_offline = {}
    rotations_offline = {}

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    timestamp_str_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_filename = os.path.join(DATA_OUTPUT_DIRECTORY, f"{base_name}_tracking_6dof_{timestamp_str_file}.csv")
    annotated_video_filename_only = f"{base_name}_annotated_{timestamp_str_file}{VIDEO_EXTENSION}"

    annotated_video_writer = get_video_writer((frame_width, frame_height), annotated_video_filename_only, fps=video_fps_original)
    # ... (CSV setup as before) ...
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['Date', 'Time', 'Marker_ID']
        if USE_REFERENCE_MARKER: # Global config determines header structure
            header.extend(['Ref_Marker_Detected', 'Pos_X', 'Pos_Y', 'Pos_Z',
                           'Rot_X', 'Rot_Y', 'Rot_Z',
                           'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z',
                           'Cam_Rot_X', 'Cam_Rot_Y', 'Cam_Rot_Z'])
        else:
            header.extend(['Pos_X', 'Pos_Y', 'Pos_Z', 'Rot_X', 'Rot_Y', 'Rot_Z'])
        csv_writer.writerow(header)

    print(f"Offline processing. Reference marker logic for this run: {'ON' if use_ref_marker_override else 'OFF'} (ID: {REFERENCE_MARKER_ID})")
    # ... (print log/video paths)

    frame_count = 0
    processing_start_time = time.time()
    ref_marker_detected_status = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        current_video_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_video_time_ms == 0 and frame_count > 1: # Fallback if POS_MSEC returns 0 after first frame
             current_video_time_ms = (frame_count -1) * (1000.0 / video_fps_original)

        date_str_log = datetime.now().strftime("%Y-%m-%d")
        
        vid_total_seconds = current_video_time_ms / 1000.0
        vid_hr = int(vid_total_seconds / 3600)
        vid_min = int((vid_total_seconds % 3600) / 60)
        vid_sec = int(vid_total_seconds % 60)
        # Corrected ms calculation from total ms
        vid_ms = int(current_video_time_ms % 1000)
        time_str_log = f"{vid_hr:02}:{vid_min:02}:{vid_sec:02}.{vid_ms:03}"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        annotated_frame, log_entries, positions_offline, rotations_offline, ref_marker_detected_status = _process_frame_common(
            gray, frame, camera_matrix, dist_coeffs, aruco_dict_obj, parameters_obj,
            MARKER_SIZE_METERS, use_ref_marker_override, REFERENCE_MARKER_ID,
            positions_offline, rotations_offline, TRAJECTORY_LENGTH, 
            date_str_log, time_str_log
        )

        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for entry in log_entries: csv_writer.writerow(entry)
        
        if use_ref_marker_override:
            status_text = f"Ref marker {REFERENCE_MARKER_ID}: {'Detected' if ref_marker_detected_status else 'Not Detected'}"
            cv2.putText(annotated_frame, status_text,
                        (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        REFERENCE_MARKER_COLOR if ref_marker_detected_status else (0, 0, 255), 2)

        cv2.putText(annotated_frame, f"Vid Time: {time_str_log}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TIMESTAMP_COLOR, 2)
        
        progress_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(annotated_frame, progress_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, OFFLINE_PROGRESS_COLOR, 2)

        if annotated_video_writer:
            annotated_video_writer.write(annotated_frame)

        if SHOW_OFFLINE_PROCESSING_PREVIEW:
            display_resized = cv2.resize(annotated_frame, 
                                         (int(frame_width * OFFLINE_PREVIEW_RESIZE_FACTOR), 
                                          int(frame_height * OFFLINE_PREVIEW_RESIZE_FACTOR)))
            cv2.imshow('Offline Processing Preview', display_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Offline processing interrupted by user.")
                break
        
        if frame_count % 100 == 0 or frame_count == total_frames:
             elapsed_time = time.time() - processing_start_time
             fps_proc = frame_count / elapsed_time if elapsed_time > 0 else 0
             progress_msg = f"Processed frame {frame_count}/{total_frames} ({fps_proc:.2f} frames/sec)"
             print(progress_msg)
             if gui_status_label: gui_status_label.config(text=progress_msg)
    
    cap.release()
    if annotated_video_writer: annotated_video_writer.release()
    if SHOW_OFFLINE_PROCESSING_PREVIEW: cv2.destroyAllWindows()

    total_processing_time = time.time() - processing_start_time
    final_msg_console = f"Offline processing finished in {total_processing_time:.2f} seconds."
    print(final_msg_console)
    print(f"Log data saved to: {csv_filename}")
    if annotated_video_writer:
        full_annotated_video_path = os.path.join(DATA_OUTPUT_DIRECTORY, VIDEO_DIRECTORY, annotated_video_filename_only)
        print(f"Annotated video saved to: {full_annotated_video_path}")
        if gui_status_label: gui_status_label.config(text=f"Done. CSV: {os.path.basename(csv_filename)}, Video: {os.path.basename(full_annotated_video_path)}")
    else:
        if gui_status_label: gui_status_label.config(text=f"Done. CSV: {os.path.basename(csv_filename)}. Video writer failed.")

# ... (calibrate_camera, generate_aruco_marker, create_chessboard - largely unchanged but ensure they use global configs)

def calibrate_camera():
    # (Ensure it uses global CALIBRATION_CHESSBOARD_SHAPE, CHESSBOARD_SQUARE_SIZE_MM correctly)
    # This function seems fine from previous versions.
    print("Camera calibration - capture multiple images of a chessboard pattern.")
    print("Press 'c' to capture, 'r' to toggle reversal, 'q' to quit and calibrate.")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    
    reverse_display = REVERSE_CAMERA_DISPLAY
    chessboard_size = CALIBRATION_CHESSBOARD_SHAPE # Use global
    square_size = CHESSBOARD_SQUARE_SIZE_MM / 1000.0 # Use global
    
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    objpoints = []
    imgpoints = []
    gray_shape_for_calib = None 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image for calibration.")
            break
        
        process_frame = frame.copy()
        display_frame = frame.copy()
        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        if gray_shape_for_calib is None: gray_shape_for_calib = gray.shape[::-1]


        ret_chess, corners_chess = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret_chess:
            cv2.drawChessboardCorners(display_frame, chessboard_size, corners_chess, ret_chess)
            cv2.putText(display_frame, "Chessboard detected! Press 'c' to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Captured: {len(objpoints)} images", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Reversal: {'ON' if reverse_display else 'OFF'} (r)", 
                    (10, display_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if reverse_display:
            display_frame = cv2.flip(display_frame, 1)
        
        cv2.imshow('Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reverse_display = not reverse_display
        elif key == ord('c') and ret_chess:
            corners2 = cv2.cornerSubPix(gray, corners_chess, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"Captured image {len(objpoints)}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(objpoints) > 3 and gray_shape_for_calib is not None: 
        print("Calculating camera calibration...")
        ret_cal, camera_matrix_cal, dist_coeffs_cal, rvecs, tvecs = cv2.calibrateCamera( # Renamed output vars
            objpoints, imgpoints, gray_shape_for_calib, None, None) 
        
        print("\nCamera Matrix:\n", camera_matrix_cal)
        print("\nDistortion Coefficients:\n", dist_coeffs_cal)
        
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                            camera_matrix_cal, dist_coeffs_cal)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print(f"Total reprojection error: {mean_error/len(objpoints)}")
        
        if not os.path.exists(DATA_OUTPUT_DIRECTORY):
            os.makedirs(DATA_OUTPUT_DIRECTORY)
        
        calibration_path = os.path.join(DATA_OUTPUT_DIRECTORY, CALIBRATION_FILE)
        np.savez(calibration_path, camera_matrix=camera_matrix_cal, dist_coeffs=dist_coeffs_cal)
        print(f"Calibration data saved to '{calibration_path}'")
        return camera_matrix_cal, dist_coeffs_cal
    else:
        print("Not enough images captured or image data error. Calibration failed.")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

def generate_aruco_marker(marker_id, size_px=600, border_px=50, save=True):
    aruco_dict_gen = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    marker_image_gray = aruco.generateImageMarker(aruco_dict_gen, marker_id, size_px)

    total_size = size_px + 2 * border_px
    # Create a BGR image from the start to allow colored drawings
    marker_with_border_color = np.ones((total_size, total_size, 3), dtype=np.uint8) * 255
    
    # Convert grayscale marker to BGR and place it
    marker_image_bgr = cv2.cvtColor(marker_image_gray, cv2.COLOR_GRAY2BGR)
    marker_with_border_color[border_px:border_px + size_px, border_px:border_px + size_px] = marker_image_bgr

    # --- Text and Drawing Properties ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color_black = (0, 0, 0) # Black for general text

    # --- Add Marker ID Text ---
    id_text = f"ID: {marker_id}"
    font_scale_id = 0.7
    font_thickness_id = 2
    (id_text_width, id_text_height), _ = cv2.getTextSize(id_text, font, font_scale_id, font_thickness_id)
    
    id_text_margin = max(5, int(border_px * 0.2)) 
    if border_px < 10: 
        id_text_margin = 2 # Minimal margin if border is tiny

    id_text_x = total_size - id_text_width - id_text_margin
    id_text_y = total_size - id_text_margin 
    cv2.putText(marker_with_border_color, id_text, (id_text_x, id_text_y), font, font_scale_id, text_color_black, font_thickness_id)

    # --- Add Axis Representation (only if border is reasonably sized) ---
    min_border_for_axes = 20 
    if border_px >= min_border_for_axes:
        axis_thickness = 2
        arrow_shaft_length_factor = 0.6 
        arrow_tip_length_factor = 0.3 # Relative to arrow shaft length

        arrow_length = int(border_px * arrow_shaft_length_factor)
        if arrow_length < 10: arrow_length = 10 # Min arrow length

        # Axis colors (X:Red, Y:Green, Z:Blue)
        color_x = (0, 0, 255) # Red BGR
        color_y = (0, 255, 0) # Green BGR
        color_z = (255, 0, 0) # Blue BGR

        font_scale_axis = 0.5
        font_thickness_axis = 1 
        label_padding = max(3, int(border_px * 0.1))

        marker_pattern_tl_x = border_px
        marker_pattern_tl_y = border_px

        # X-axis (points right, in bottom border area)
        x_arrow_start_x = marker_pattern_tl_x
        x_arrow_start_y = marker_pattern_tl_y + size_px + border_px // 2 
        x_arrow_end_x = x_arrow_start_x + arrow_length
        x_arrow_end_y = x_arrow_start_y
        cv2.arrowedLine(marker_with_border_color, (x_arrow_start_x, x_arrow_start_y), (x_arrow_end_x, x_arrow_end_y),
                        color_x, axis_thickness, tipLength=arrow_tip_length_factor)
        (x_label_w, x_label_h), _ = cv2.getTextSize("X", font, font_scale_axis, font_thickness_axis)
        cv2.putText(marker_with_border_color, "X",
                    (x_arrow_end_x + label_padding, x_arrow_start_y + x_label_h // 2),
                    font, font_scale_axis, color_x, font_thickness_axis)

        # Y-axis (points down, in right border area)
        y_arrow_start_x = marker_pattern_tl_x + size_px + border_px // 2 
        y_arrow_start_y = marker_pattern_tl_y
        y_arrow_end_x = y_arrow_start_x
        y_arrow_end_y = y_arrow_start_y + arrow_length
        cv2.arrowedLine(marker_with_border_color, (y_arrow_start_x, y_arrow_start_y), (y_arrow_end_x, y_arrow_end_y),
                        color_y, axis_thickness, tipLength=arrow_tip_length_factor)
        (y_label_w, y_label_h), _ = cv2.getTextSize("Y", font, font_scale_axis, font_thickness_axis)
        cv2.putText(marker_with_border_color, "Y",
                    (y_arrow_start_x - y_label_w // 2 , y_arrow_end_y + y_label_h + label_padding), 
                    font, font_scale_axis, color_y, font_thickness_axis)

        # Z-axis (points out of the plane - Symbol in top-left border area)
        z_symbol_center_x = border_px // 2
        z_symbol_center_y = border_px // 2
        z_circle_radius = max(5, border_px // 4) 

        cv2.circle(marker_with_border_color, (z_symbol_center_x, z_symbol_center_y), z_circle_radius, color_z, axis_thickness)
        cv2.circle(marker_with_border_color, (z_symbol_center_x, z_symbol_center_y), max(1, z_circle_radius // 3), color_z, -1) # Dot
        
        (z_label_main_w, z_label_main_h), _ = cv2.getTextSize("Z", font, font_scale_axis, font_thickness_axis)
        cv2.putText(marker_with_border_color, "Z", 
                    (z_symbol_center_x + z_circle_radius + label_padding, z_symbol_center_y + z_label_main_h // 2),
                    font, font_scale_axis, color_z, font_thickness_axis)
        
        (z_label_out_w, z_label_out_h), _ = cv2.getTextSize("(out)", font, font_scale_axis * 0.8, font_thickness_axis)
        cv2.putText(marker_with_border_color, "(out)", 
                    (z_symbol_center_x + z_circle_radius + label_padding, z_symbol_center_y + z_label_main_h // 2 + z_label_out_h + 2),
                    font, font_scale_axis * 0.8, color_z, font_thickness_axis)
    else:
        if border_px > 0 : # Only print warning if border exists but is too small
            print(f"Note: Marker border ({border_px}px) is too small for axis representation (min {min_border_for_axes}px recommended). Axes not drawn.")


    # --- Add "REFERENCE MARKER" Text if applicable (top border, centered) ---
    if marker_id == REFERENCE_MARKER_ID and USE_REFERENCE_MARKER:
        ref_text = "REFERENCE MARKER"
        font_scale_ref = 0.8 
        font_thickness_ref = 2
        (text_width_ref, text_height_ref), _ = cv2.getTextSize(ref_text, font, font_scale_ref, font_thickness_ref)
        
        ref_text_x = (total_size - text_width_ref) // 2
        ref_text_y = border_px // 2 + text_height_ref // 2 
        cv2.putText(marker_with_border_color, ref_text, (text_x_ref, ref_text_y), font, font_scale_ref, text_color_black, font_thickness_ref)

    if save:
        if not os.path.exists(DATA_OUTPUT_DIRECTORY):
            os.makedirs(DATA_OUTPUT_DIRECTORY)
        
        ref_tag_filename = "_REFERENCE" if (marker_id == REFERENCE_MARKER_ID and USE_REFERENCE_MARKER) else ""
        filename = os.path.join(DATA_OUTPUT_DIRECTORY, f"aruco_marker_id{marker_id}{ref_tag_filename}_with_axes.png")
        cv2.imwrite(filename, marker_with_border_color)
        print(f"Marker with ID and axes representation saved as {filename}")
    
    return marker_with_border_color

def create_chessboard(squares_x, squares_y, square_size_px=100, save=True):
    # This function seems fine, uses global CHESSBOARD_SQUARE_SIZE_MM
    width = (squares_x + 1) * square_size_px 
    height = (squares_y + 1) * square_size_px
    chessboard = np.ones((height, width), dtype=np.uint8) * 255
    
    for y_idx in range(squares_y + 1):
        for x_idx in range(squares_x + 1):
            if (x_idx + y_idx) % 2 == 0:
                y0, y1 = y_idx * square_size_px, (y_idx + 1) * square_size_px
                x0, x1 = x_idx * square_size_px, (x_idx + 1) * square_size_px
                chessboard[y0:y1, x0:x1] = 0
    
    instructions_height = 100
    instructions = np.ones((instructions_height, width), dtype=np.uint8) * 255
    text1 = "Camera Calibration Pattern"
    (tw1, th1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(instructions, text1, ((width - tw1) // 2, instructions_height // 3), # Adjusted y
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    
    corners_x_disp, corners_y_disp = squares_x, squares_y 
    text2 = f"{corners_x_disp}x{corners_y_disp} internal corners, square size: {CHESSBOARD_SQUARE_SIZE_MM}mm"
    (tw2, th2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(instructions, text2, ((width - tw2) // 2, instructions_height * 2 // 3 + th2 //2 ), # Adjusted y
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    final_image = np.vstack((instructions, chessboard))
    
    if save:
        if not os.path.exists(DATA_OUTPUT_DIRECTORY): os.makedirs(DATA_OUTPUT_DIRECTORY)
        filename = os.path.join(DATA_OUTPUT_DIRECTORY, f"chessboard_{corners_x_disp}x{corners_y_disp}.png")
        cv2.imwrite(filename, final_image)
        print(f"Chessboard pattern saved as {filename}")
    return final_image

#################################
# GUI for Offline Processing
#################################
class OfflineProcessingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Offline Video Processor")

        self.label = ttk.Label(master, text="Select a video file for offline ArUco processing.")
        self.label.pack(pady=10)

        self.filepath_var = tk.StringVar()
        self.filepath_entry = ttk.Entry(master, textvariable=self.filepath_var, width=50, state="readonly")
        self.filepath_entry.pack(pady=5, padx=10, side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_button = ttk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5, padx=5, side=tk.LEFT)

        self.ref_marker_var = tk.BooleanVar(value=USE_REFERENCE_MARKER) # Default to global
        self.ref_marker_check = ttk.Checkbutton(master, text=f"Use Reference Marker (ID: {REFERENCE_MARKER_ID})",
                                               variable=self.ref_marker_var)
        self.ref_marker_check.pack(pady=5)
        
        self.process_button = ttk.Button(master, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.status_label = ttk.Label(master, text="")
        self.status_label.pack(pady=10)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", "*.mp4"),
                       ("AVI files", "*.avi"), 
                       ("All files", "*.*"))
        )
        if filename:
            self.filepath_var.set(filename)
            self.process_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Selected: {os.path.basename(filename)}")

    def start_processing(self):
        video_path = self.filepath_var.get()
        if not video_path:
            messagebox.showerror("Error", "No video file selected.")
            return

        use_ref = self.ref_marker_var.get()
        
        # Disable button during processing (GUI might still freeze if not threaded)
        self.process_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.status_label.config(text="Processing... Check console for progress.")
        self.master.update_idletasks() # Ensure GUI updates before blocking call

        try:
            # Note: This call will block the GUI. For true responsiveness,
            # process_video_offline should be run in a separate thread.
            process_video_offline(video_path, use_ref, self.status_label)
            # Status label is updated inside process_video_offline upon completion/error
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.status_label.config(text=f"Error: {e}")
        finally:
            self.process_button.config(state=tk.NORMAL if self.filepath_var.get() else tk.DISABLED)
            self.browse_button.config(state=tk.NORMAL)
            # Final status message is set by process_video_offline or here if error caught early

def launch_offline_gui():
    root = tk.Tk()
    # Optional: Add some styling if ttk is used
    # style = ttk.Style(root)
    # style.theme_use('clam') # Or 'alt', 'default', 'classic'
    gui = OfflineProcessingGUI(root)
    root.mainloop()

#################################
# MAIN MENU & EXECUTION
#################################
def display_menu():
    print("\n=== Buoy Tracking with ArUco Markers ===")
    print("1. Generate ArUco marker")
    print("2. Generate reference marker (uses configured ID)")
    print("3. Generate chessboard calibration pattern")
    print("4. Calibrate camera")
    print("5. Start buoy tracking (Live Mode)")
    print("6. Launch Offline Video Processor GUI") # Updated
    print("7. Configure settings (CLI - Basic)") # Clarified
    print("8. Exit")
    return input("Enter your choice (1-8): ")

def configure_settings_cli():
    global USE_REFERENCE_MARKER, REFERENCE_MARKER_ID, RECORD_VIDEO, REVERSE_CAMERA_DISPLAY
    print("\n--- Basic CLI Configuration ---")
    
    # Reference Marker
    current_ref_status = 'ON' if USE_REFERENCE_MARKER else 'OFF'
    use_ref_str = input(f"Use reference marker? (y/n, current: {current_ref_status}): ").lower()
    if use_ref_str == 'y': USE_REFERENCE_MARKER = True
    elif use_ref_str == 'n': USE_REFERENCE_MARKER = False

    if USE_REFERENCE_MARKER:
        try:
            new_ref_id = input(f"Reference marker ID (current: {REFERENCE_MARKER_ID}): ")
            if new_ref_id: REFERENCE_MARKER_ID = int(new_ref_id)
        except ValueError: print("Invalid ID, keeping current.")

    # Record Video Default (Live)
    current_rec_status = 'ON' if RECORD_VIDEO else 'OFF'
    rec_str = input(f"Record video by default (Live Mode)? (y/n, current: {current_rec_status}): ").lower()
    if rec_str == 'y': RECORD_VIDEO = True
    elif rec_str == 'n': RECORD_VIDEO = False
    
    # Reverse Display Default (Live)
    current_rev_status = 'ON' if REVERSE_CAMERA_DISPLAY else 'OFF'
    rev_str = input(f"Reverse camera display by default (Live Mode)? (y/n, current: {current_rev_status}): ").lower()
    if rev_str == 'y': REVERSE_CAMERA_DISPLAY = True
    elif rev_str == 'n': REVERSE_CAMERA_DISPLAY = False
    
    print("Settings updated.")


if __name__ == "__main__":
    if not os.path.exists(DATA_OUTPUT_DIRECTORY):
        os.makedirs(DATA_OUTPUT_DIRECTORY)
    
    video_output_path = os.path.join(DATA_OUTPUT_DIRECTORY, VIDEO_DIRECTORY)
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path)
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            # ... (generate ArUco marker - logic from previous version)
            try:
                marker_id_input = int(input(f"Enter marker ID (e.g., 0-{aruco.getPredefinedDictionary(ARUCO_DICT_TYPE).bytesList.shape[0]-1}): "))
                marker_size_px_input = int(input("Enter marker size in pixels (default: 600): ") or "600")
                marker_img_disp = generate_aruco_marker(marker_id_input, marker_size_px_input)
                cv2.imshow(f"ArUco Marker ID: {marker_id_input}", marker_img_disp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except ValueError: print("Invalid input.")
            
        elif choice == '2':
            # ... (generate reference marker - logic from previous version)
            try:
                marker_size_px_input = int(input(f"Enter marker size for Reference ID {REFERENCE_MARKER_ID} (default: 600): ") or "600")
                marker_img_disp = generate_aruco_marker(REFERENCE_MARKER_ID, marker_size_px_input) # Uses global REFERENCE_MARKER_ID
                cv2.imshow(f"Reference Marker ID: {REFERENCE_MARKER_ID}", marker_img_disp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except ValueError: print("Invalid input.")

        elif choice == '3':
            # ... (generate chessboard - logic from previous version)
            try:
                squares_x_conf = CALIBRATION_CHESSBOARD_SHAPE[0] # Use configured
                squares_y_conf = CALIBRATION_CHESSBOARD_SHAPE[1]
                print(f"Generating chessboard with {squares_x_conf}x{squares_y_conf} internal corners.")
                square_size_px_disp_input = int(input("Enter square size in pixels for display (default: 100): ") or "100")
                chessboard_img_disp = create_chessboard(squares_x_conf, squares_y_conf, square_size_px_disp_input)
                cv2.imshow("Chessboard Pattern", chessboard_img_disp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except ValueError: print("Invalid input.")

        elif choice == '4':
            calibrate_camera()
            
        elif choice == '5': # Live Tracking
            print("\n--- Live Tracking Setup ---")
            # Confirm live settings (using globals that might have been changed by config)
            print(f"Using Reference Marker ID {REFERENCE_MARKER_ID}: {'YES' if USE_REFERENCE_MARKER else 'NO'}")
            print(f"Record Video by default: {'YES' if RECORD_VIDEO else 'NO'}")
            print(f"Reverse Camera Display by default: {'YES' if REVERSE_CAMERA_DISPLAY else 'NO'}")
            if input("Proceed with these settings? (y/n, default y): ").lower() == 'n':
                print("Live tracking cancelled. Configure settings via option 7 if needed.")
                continue
            track_buoy_with_aruco_6dof()

        elif choice == '6': # Launch Offline GUI
            print("Launching Offline Video Processor GUI...")
            launch_offline_gui()
            
        elif choice == '7':
            configure_settings_cli() # Call the CLI config
            
        elif choice == '8':
            print("Exiting program.")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")