import cv2
import numpy as np
from cv2 import aruco
import time
import csv
import os
from datetime import datetime

#################################
# CONFIGURATION PARAMETERS
#################################

# Camera Setup
CAMERA_INDEX = 0                  # Camera index (0 for default webcam)
CAMERA_RESOLUTION = (3840, 2160)  # Camera resolution (width, height)
CAMERA_FPS = 30                   # Camera frames per second
REVERSE_CAMERA_DISPLAY = True     # Whether to horizontally flip the camera display (mirror mode)

# Video Recording Settings
RECORD_VIDEO = False              # Whether to record video by default
VIDEO_CODEC = 'MJPG'              # Video codec for recording: XVID, MJPG, H264 (if available)
VIDEO_FPS = 30                    # FPS for recorded video

# ArUco Marker Configuration
ARUCO_DICT_TYPE = aruco.DICT_6X6_250  # ArUco dictionary type
MARKER_SIZE_METERS = 0.16            # Size of ArUco marker in meters (5 cm)

# Reference Marker Configuration
REFERENCE_MARKER_ID = 23# ID of the marker to use as reference frame
USE_REFERENCE_MARKER = True       # Whether to use a reference marker coordinate system

# Camera Calibration Parameters (default values - should be replaced with actual calibration)
# These are example values - use camera_calibration() to get accurate values
DEFAULT_CAMERA_MATRIX = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

DEFAULT_DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)  # Assuming no lens distortion

# Calibration Settings
CALIBRATION_CHESSBOARD_SHAPE = (9, 6)  # Number of internal corners (width, height)
CHESSBOARD_SQUARE_SIZE_MM = 26.5        # Size of each chessboard square in mm

# Visualization Settings
TRAJECTORY_LENGTH = 30                 # Number of positions to keep for trajectory display
TRAJECTORY_COLOR = (0, 0, 255)         # Color of trajectory line (BGR)
MARKER_INFO_COLOR = (0, 255, 0)        # Color of marker info text (BGR)
TIMESTAMP_COLOR = (0, 0, 255)          # Color of timestamp display (BGR)
REFERENCE_MARKER_COLOR = (255, 0, 255) # Color for reference marker (purple)
RECORD_INDICATOR_COLOR = (0, 0, 255)   # Color for recording indicator (red)

# File Settings
DATA_OUTPUT_DIRECTORY = "buoy_tracking_data"  # Directory to save data files
CALIBRATION_FILE = "camera_calibration.npz"   # File to save/load camera calibration
VIDEO_DIRECTORY = "recordings"               # Directory to save video recordings

#################################
# UTILITY FUNCTIONS
#################################

def transform_pose_to_reference(marker_rvec, marker_tvec, ref_rvec, ref_tvec):
    """
    Transform marker pose to reference marker coordinate system
    
    Parameters:
    marker_rvec: Rotation vector of the marker
    marker_tvec: Translation vector of the marker
    ref_rvec: Rotation vector of the reference marker
    ref_tvec: Translation vector of the reference marker
    
    Returns:
    rel_rvec: Rotation vector relative to reference marker
    rel_tvec: Translation vector relative to reference marker
    """
    # Convert rotation vectors to rotation matrices
    ref_rmat, _ = cv2.Rodrigues(ref_rvec)
    marker_rmat, _ = cv2.Rodrigues(marker_rvec)
    
    # Compute the inverse of the reference rotation matrix
    ref_rmat_inv = np.transpose(ref_rmat)
    
    # Compute relative translation
    rel_tvec = np.dot(ref_rmat_inv, marker_tvec - ref_tvec)
    
    # Compute relative rotation matrix
    rel_rmat = np.dot(ref_rmat_inv, marker_rmat)
    
    # Convert back to rotation vector
    rel_rvec, _ = cv2.Rodrigues(rel_rmat)
    
    return rel_rvec, rel_tvec

def rotation_vector_to_euler(rvec):
    """Convert rotation vector to Euler angles (degrees)"""
    rmat, _ = cv2.Rodrigues(rvec)
    euler_angles = cv2.decomposeProjectionMatrix(
        np.hstack((rmat, np.zeros((3, 1)))))[6]
    return euler_angles

def load_camera_calibration():
    """Load camera calibration parameters if they exist"""
    calibration_path = os.path.join(DATA_OUTPUT_DIRECTORY, CALIBRATION_FILE)
    try:
        data = np.load(calibration_path)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print(f"Loaded camera calibration parameters from {calibration_path}")
        return camera_matrix, dist_coeffs
    except:
        print("No calibration file found. Using default parameters.")
        # Default parameters (not accurate!)
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

def get_video_writer(frame_size, filename):
    """Create a VideoWriter object for recording"""
    # Create video directory if it doesn't exist
    video_path = os.path.join(DATA_OUTPUT_DIRECTORY, VIDEO_DIRECTORY)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    
    # Full path for video file
    video_file = os.path.join(video_path, filename)
    
    # Get the four-character code for the specified codec
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    
    # Create VideoWriter object
    writer = cv2.VideoWriter(video_file, fourcc, VIDEO_FPS, frame_size)
    
    if not writer.isOpened():
        print(f"Warning: Could not initialize video writer with codec {VIDEO_CODEC}.")
        print("Trying fallback codec MJPG...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(video_file, fourcc, VIDEO_FPS, frame_size)
        
        if not writer.isOpened():
            print("Error: Failed to create video writer. Video recording disabled.")
            return None
    
    print(f"Video will be saved to: {video_file}")
    return writer

#################################
# MAIN FUNCTIONS
#################################

def track_buoy_with_aruco_6dof():
    """Track buoys with ArUco markers and record 6DOF data"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(DATA_OUTPUT_DIRECTORY):
        os.makedirs(DATA_OUTPUT_DIRECTORY)
    
    # Initialize the webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Set camera resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Load camera calibration if available, otherwise use default values
    camera_matrix, dist_coeffs = load_camera_calibration()
    
    # Set up ArUco dictionary and parameters
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = aruco.DetectorParameters()
    
    # Variables to store tracking data
    positions = {}  # Dictionary to store positions by marker ID
    rotations = {}  # Dictionary to store rotations by marker ID
    timestamps = []
    
    # Generate timestamp for filenames
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a CSV file to save the 6DOF data
    csv_filename = os.path.join(DATA_OUTPUT_DIRECTORY, 
                              f"buoy_tracking_6dof_{timestamp_str}.csv")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header based on reference frame mode
        if USE_REFERENCE_MARKER:
            csv_writer.writerow(['Date', 'Time', 'Marker_ID', 
                                'Ref_Marker_Detected',
                                'Pos_X', 'Pos_Y', 'Pos_Z', 
                                'Rot_X', 'Rot_Y', 'Rot_Z',
                                'Cam_Pos_X', 'Cam_Pos_Y', 'Cam_Pos_Z', 
                                'Cam_Rot_X', 'Cam_Rot_Y', 'Cam_Rot_Z'])
        else:
            csv_writer.writerow(['Date', 'Time', 'Marker_ID', 
                                'Pos_X', 'Pos_Y', 'Pos_Z', 
                                'Rot_X', 'Rot_Y', 'Rot_Z'])
    
    # Setup video recording if enabled
    record_video = RECORD_VIDEO
    video_writer = None
    video_filename = f"buoy_tracking_{timestamp_str}.avi"
    
    # Read one frame to get the frame size
    ret, first_frame = cap.read()
    if ret:
        frame_size = (first_frame.shape[1], first_frame.shape[0])
        if record_video:
            video_writer = get_video_writer(frame_size, video_filename)
            if video_writer is None:
                record_video = False
    
    print(f"Starting buoy tracking. Data will be saved to {csv_filename}. Press 'q' to quit.")
    print(f"Reference marker ID: {REFERENCE_MARKER_ID}" if USE_REFERENCE_MARKER else "Using camera as reference frame")
    print(f"Camera display reversed: {REVERSE_CAMERA_DISPLAY}")
    print(f"Video recording: {'ON' if record_video else 'OFF'}")
    print("Controls:")
    print("  'q' - Quit")
    print("  'r' - Toggle camera reversal")
    print("  'v' - Toggle video recording")
    
    # Camera reverse mode flag (can be toggled during runtime)
    reverse_display = REVERSE_CAMERA_DISPLAY
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Create a copy of the original frame for processing
        # This ensures we're doing marker detection on the original frame
        process_frame = frame.copy()
        
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        # Get current timestamp with millisecond precision
        now = datetime.now()
        unix_timestamp = time.time()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S.%f")[:-3]  # Format with milliseconds (HH:MM:SS.mmm)
        
        # Variables for reference marker
        ref_marker_detected = False
        ref_rvec = None
        ref_tvec = None
        
        # Create a display frame (this might be flipped)
        display_frame = process_frame.copy()
        
        # If markers are detected
        if ids is not None and len(ids) > 0:
            # Draw detected markers on the display frame
            aruco.drawDetectedMarkers(display_frame, corners, ids)
            
            # Estimate pose for each detected marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE_METERS, camera_matrix, dist_coeffs)
            
            # First, look for reference marker if enabled
            if USE_REFERENCE_MARKER:
                for i, marker_id in enumerate(ids):
                    if marker_id[0] == REFERENCE_MARKER_ID:
                        ref_marker_detected = True
                        ref_rvec = rvecs[i][0]
                        ref_tvec = tvecs[i][0]
                        
                        # Draw reference marker with special color
                        cv2.drawFrameAxes(display_frame, camera_matrix, dist_coeffs, 
                                         ref_rvec, ref_tvec, MARKER_SIZE_METERS, 3)
                        
                        # Label as reference
                        c = corners[i][0]
                        center_px = (int(np.mean(c[:, 0])), int(np.mean(c[:, 1])))
                        cv2.putText(display_frame, "REFERENCE", 
                                    (center_px[0] - 40, center_px[1] - 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_MARKER_COLOR, 2)
                        break
            
            # Process all markers
            for i, marker_id in enumerate(ids):
                # Skip reference marker in second pass
                if USE_REFERENCE_MARKER and marker_id[0] == REFERENCE_MARKER_ID:
                    continue
                    
                # Get marker pose in camera coordinates
                marker_rvec = rvecs[i][0]
                marker_tvec = tvecs[i][0]
                
                # Calculate pose relative to reference marker if available
                if ref_marker_detected and marker_id[0] != REFERENCE_MARKER_ID:
                    rel_rvec, rel_tvec = transform_pose_to_reference(
                        marker_rvec, marker_tvec, ref_rvec, ref_tvec)
                else:
                    rel_rvec, rel_tvec = marker_rvec, marker_tvec
                
                # Draw axis for each marker
                cv2.drawFrameAxes(display_frame, camera_matrix, dist_coeffs, 
                                 marker_rvec, marker_tvec, MARKER_SIZE_METERS/2)
                
                # Get the center of the marker (in pixel coordinates)
                c = corners[i][0]
                center_px = (int(np.mean(c[:, 0])), int(np.mean(c[:, 1])))
                
                # Calculate Euler angles
                euler_angles_cam = rotation_vector_to_euler(marker_rvec)
                
                if ref_marker_detected and marker_id[0] != REFERENCE_MARKER_ID:
                    euler_angles_rel = rotation_vector_to_euler(rel_rvec)
                else:
                    euler_angles_rel = euler_angles_cam
                
                # Store position and rotation data
                marker_id_val = marker_id[0]
                if marker_id_val not in positions:
                    positions[marker_id_val] = []
                    rotations[marker_id_val] = []
                
                positions[marker_id_val].append(center_px)
                rotations[marker_id_val].append(euler_angles_rel)
                
                # Keep only the last N positions for each marker
                if len(positions[marker_id_val]) > TRAJECTORY_LENGTH:
                    positions[marker_id_val].pop(0)
                    rotations[marker_id_val].pop(0)
                
                # Save data to CSV
                with open(csv_filename, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    
                    if USE_REFERENCE_MARKER:
                        csv_writer.writerow([
                            date_str, 
                            time_str,
                            marker_id_val,
                            "1" if ref_marker_detected else "0",
                            f"{rel_tvec[0]:.6f}", f"{rel_tvec[1]:.6f}", f"{rel_tvec[2]:.6f}",
                            f"{euler_angles_rel[0][0]:.3f}", f"{euler_angles_rel[1][0]:.3f}", f"{euler_angles_rel[2][0]:.3f}",
                            f"{marker_tvec[0]:.6f}", f"{marker_tvec[1]:.6f}", f"{marker_tvec[2]:.6f}",
                            f"{euler_angles_cam[0][0]:.3f}", f"{euler_angles_cam[1][0]:.3f}", f"{euler_angles_cam[2][0]:.3f}"
                        ])
                    else:
                        csv_writer.writerow([
                            date_str, 
                            time_str,
                            marker_id_val,
                            f"{marker_tvec[0]:.6f}", f"{marker_tvec[1]:.6f}", f"{marker_tvec[2]:.6f}",
                            f"{euler_angles_cam[0][0]:.3f}", f"{euler_angles_cam[1][0]:.3f}", f"{euler_angles_cam[2][0]:.3f}"
                        ])
                
                # Display marker ID
                cv2.putText(display_frame, f"ID: {marker_id_val}", 
                            (center_px[0] + 10, center_px[1] - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARKER_INFO_COLOR, 2)
                
                # Display position
                if ref_marker_detected and marker_id_val != REFERENCE_MARKER_ID:
                    cv2.putText(display_frame, f"Rel Pos: ({rel_tvec[0]:.3f}, {rel_tvec[1]:.3f}, {rel_tvec[2]:.3f})m", 
                                (center_px[0] + 10, center_px[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARKER_INFO_COLOR, 2)
                else:
                    cv2.putText(display_frame, f"Pos: ({marker_tvec[0]:.3f}, {marker_tvec[1]:.3f}, {marker_tvec[2]:.3f})m", 
                                (center_px[0] + 10, center_px[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARKER_INFO_COLOR, 2)
                
                # Display rotation
                rot_text = "Rel Rot" if (ref_marker_detected and marker_id_val != REFERENCE_MARKER_ID) else "Rot"
                cv2.putText(display_frame, 
                            f"{rot_text}: ({euler_angles_rel[0][0]:.1f}, {euler_angles_rel[1][0]:.1f}, {euler_angles_rel[2][0]:.1f})°", 
                            (center_px[0] + 10, center_px[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, MARKER_INFO_COLOR, 2)
        
        # Draw trajectory for each marker
        for marker_id_val, pos_list in positions.items():
            if len(pos_list) > 1:
                for i in range(1, len(pos_list)):
                    cv2.line(display_frame, pos_list[i-1], pos_list[i], TRAJECTORY_COLOR, 2)
        
        # Add status information to the display frame
        # Reference frame status
        status_text = f"Ref marker {REFERENCE_MARKER_ID}: {'Detected' if ref_marker_detected else 'Not Detected'}"
        cv2.putText(display_frame, status_text, 
                    (10, display_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    REFERENCE_MARKER_COLOR if ref_marker_detected else (0, 0, 255), 2)
        
        # Camera reversal status
        cv2.putText(display_frame, f"Reversal: {'ON' if reverse_display else 'OFF'} (r)", 
                    (10, display_frame.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Display time on frame with millisecond precision
        cv2.putText(display_frame, f"{date_str} {time_str}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TIMESTAMP_COLOR, 2)
        
        # Add recording indicator if recording
        if record_video:
            # Pulsing red circle as recording indicator
            radius = 15
            pulse = int(10 * np.sin(time.time() * 5) + 15)  # Pulsating size
            cv2.circle(display_frame, (display_frame.shape[1] - 30, 30), radius, RECORD_INDICATOR_COLOR, pulse)
            cv2.putText(display_frame, "REC", 
                        (display_frame.shape[1] - 80, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RECORD_INDICATOR_COLOR, 2)
            
            # Recording status
            cv2.putText(display_frame, f"Recording: ON (v)", 
                        (10, display_frame.shape[0] - 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RECORD_INDICATOR_COLOR, 2)
        else:
            # Recording status
            cv2.putText(display_frame, f"Recording: OFF (v)", 
                        (10, display_frame.shape[0] - 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Create a frame for recording - always use the display frame
        # This ensures what you see is what gets recorded
        record_frame = display_frame.copy()
        
        # Flip the display frame horizontally if in reverse mode
        if reverse_display:
            display_frame = cv2.flip(display_frame, 1)  # 1 means horizontal flip
        
        # Write the frame to video file if recording
        if record_video and video_writer is not None:
            video_writer.write(record_frame)
        
        # Display the resulting frame
        cv2.imshow('Buoy 6DOF Tracking', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit
            break
        elif key == ord('r'):
            # Toggle camera reversal
            reverse_display = not reverse_display
            print(f"Camera display reversal: {'ON' if reverse_display else 'OFF'}")
        elif key == ord('v'):
            # Toggle video recording
            if record_video:
                # Stop recording
                record_video = False
                if video_writer is not None:
                    video_writer.release()
                    print(f"Video recording stopped and saved to {video_filename}")
                    video_writer = None
            else:
                # Start recording
                record_video = True
                if video_writer is None:
                    # Generate new filename for each recording session
                    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"buoy_tracking_{timestamp_now}.avi"
                    video_writer = get_video_writer(frame_size, video_filename)
                    if video_writer is None:
                        record_video = False
                        print("Failed to start video recording.")
                    else:
                        print(f"Video recording started. Saving to {video_filename}")
    
    # Clean up and release resources
    if record_video and video_writer is not None:
        video_writer.release()
        print(f"Video recording saved to {os.path.join(DATA_OUTPUT_DIRECTORY, VIDEO_DIRECTORY, video_filename)}")
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Tracking data saved to {csv_filename}")
    return csv_filename

def calibrate_camera():
    """
    Camera calibration using a chessboard pattern.
    Returns camera_matrix and dist_coeffs.
    """
    print("Camera calibration - capture multiple images of a chessboard pattern.")
    print("Press 'c' to capture an image, 'r' to toggle camera reversal, 'q' to quit and calculate calibration.")
    
    # Initialize the webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    
    # Camera reverse mode flag
    reverse_display = REVERSE_CAMERA_DISPLAY
    
    # Chessboard dimensions
    chessboard_size = CALIBRATION_CHESSBOARD_SHAPE
    
    # Square size in meters (convert from mm)
    square_size = CHESSBOARD_SQUARE_SIZE_MM / 1000.0  
    
    # Prepare object points: (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # Scale to actual size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        # Create a copy for processing (not reversed)
        process_frame = frame.copy()
        
        # Create a display frame that might be reversed
        display_frame = frame.copy()
        
        # Convert processing frame to grayscale
        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, draw corners on the display frame
        if ret:
            # Draw the corners
            cv2.drawChessboardCorners(display_frame, chessboard_size, corners, ret)
            cv2.putText(display_frame, "Chessboard detected! Press 'c' to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(display_frame, f"Captured: {len(objpoints)} images", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display camera reversal status
        cv2.putText(display_frame, f"Camera Reversal: {'ON' if reverse_display else 'OFF'} (press 'r' to toggle)", 
                    (10, display_frame.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Flip the display frame horizontally if in reverse mode
        if reverse_display:
            display_frame = cv2.flip(display_frame, 1)
        
        cv2.imshow('Camera Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Toggle camera reversal
            reverse_display = not reverse_display
            print(f"Camera display reversal: {'ON' if reverse_display else 'OFF'}")
        elif key == ord('c') and ret:
            # Note: We use the corners from the original non-reversed frame for calibration
            # Refine corner positions for better accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"Captured image {len(objpoints)}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(objpoints) > 0:
        print("Calculating camera calibration...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("\nCamera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                            camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        print(f"Total reprojection error: {mean_error/len(objpoints)}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(DATA_OUTPUT_DIRECTORY):
            os.makedirs(DATA_OUTPUT_DIRECTORY)
        
        # Save to a file
        calibration_path = os.path.join(DATA_OUTPUT_DIRECTORY, CALIBRATION_FILE)
        np.savez(calibration_path, 
                camera_matrix=camera_matrix, 
                dist_coeffs=dist_coeffs)
        
        print(f"Calibration data saved to '{calibration_path}'")
        return camera_matrix, dist_coeffs
    else:
        print("No images captured. Calibration failed.")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

def generate_aruco_marker(marker_id, size_px=600, border_px=50, save=True):
    """
    Generate and save an ArUco marker image
    
    Parameters:
    marker_id (int): ID of the marker to generate
    size_px (int): Size of the marker in pixels
    border_px (int): Size of the white border in pixels
    save (bool): Whether to save the marker to a file
    
    Returns:
    numpy.ndarray: The generated marker image
    """

# Create dictionary and generate marker
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    
    # Add a white border
    marker_with_border = np.ones((size_px + 2*border_px, 
                                  size_px + 2*border_px), 
                                  dtype=np.uint8) * 255
    
    marker_with_border[border_px:border_px+size_px, 
                      border_px:border_px+size_px] = marker_image
    
    # Add text indicating if this is a reference marker
    if marker_id == REFERENCE_MARKER_ID and USE_REFERENCE_MARKER:
        ref_text = "REFERENCE MARKER"
        cv2.putText(marker_with_border, ref_text, 
                    (size_px//2 - 100, border_px//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    
    # Save marker if requested
    if save:
        # Create output directory if it doesn't exist
        if not os.path.exists(DATA_OUTPUT_DIRECTORY):
            os.makedirs(DATA_OUTPUT_DIRECTORY)
            
        ref_tag = "_REFERENCE" if (marker_id == REFERENCE_MARKER_ID and USE_REFERENCE_MARKER) else ""
        filename = os.path.join(DATA_OUTPUT_DIRECTORY, f"aruco_marker_id{marker_id}{ref_tag}.png")
        cv2.imwrite(filename, marker_with_border)
        print(f"Marker saved as {filename}")
    
    return marker_with_border

def create_chessboard(squares_x, squares_y, square_size_px=100, save=True):
    """
    Create a chessboard calibration pattern
    
    Parameters:
    squares_x (int): Number of squares in x direction
    squares_y (int): Number of squares in y direction
    square_size_px (int): Size of each square in pixels
    save (bool): Whether to save the pattern to a file
    
    Returns:
    numpy.ndarray: The generated chessboard image
    """
    # Create a white image
    width = (squares_x + 1) * square_size_px  # +1 for even number of black corners
    height = (squares_y + 1) * square_size_px
    chessboard = np.ones((height, width), dtype=np.uint8) * 255
    
    # Fill with chessboard pattern
    for y in range(squares_y + 1):
        for x in range(squares_x + 1):
            if (x + y) % 2 == 0:
                # Draw black square
                y0 = y * square_size_px
                y1 = (y + 1) * square_size_px
                x0 = x * square_size_px
                x1 = (x + 1) * square_size_px
                chessboard[y0:y1, x0:x1] = 0
    
    # Add instructions
    instructions = np.ones((100, width), dtype=np.uint8) * 255
    text = "Camera Calibration Pattern"
    cv2.putText(instructions, text, (width//2 - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    
    corners_x, corners_y = squares_x, squares_y
    text = f"{corners_x}x{corners_y} internal corners, square size: {CHESSBOARD_SQUARE_SIZE_MM}mm"
    cv2.putText(instructions, text, (width//2 - 200, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    # Combine image and instructions
    final_image = np.vstack((instructions, chessboard))
    
    # Save pattern if requested
    if save:
        # Create output directory if it doesn't exist
        if not os.path.exists(DATA_OUTPUT_DIRECTORY):
            os.makedirs(DATA_OUTPUT_DIRECTORY)
            
        filename = os.path.join(DATA_OUTPUT_DIRECTORY, f"chessboard_{corners_x}x{corners_y}.png")
        cv2.imwrite(filename, final_image)
        print(f"Chessboard pattern saved as {filename}")
    
    return final_image

def display_menu():
    """Display a menu of options for the user"""
    print("\n=== Buoy Tracking with ArUco Markers ===")
    print("1. Generate ArUco marker")
    print("2. Generate reference marker")
    print("3. Generate chessboard calibration pattern")
    print("4. Calibrate camera")
    print("5. Start buoy tracking")
    print("6. Configure settings")
    print("7. Exit")
    return input("Enter your choice (1-7): ")

def configure_settings():
    """Configure application settings"""
    global CAMERA_INDEX, CAMERA_RESOLUTION, CAMERA_FPS
    global REVERSE_CAMERA_DISPLAY, RECORD_VIDEO, VIDEO_CODEC, VIDEO_FPS
    global ARUCO_DICT_TYPE, MARKER_SIZE_METERS
    global REFERENCE_MARKER_ID, USE_REFERENCE_MARKER
    global TRAJECTORY_LENGTH
    
    print("\n=== Configuration Menu ===")
    print("1. Camera settings")
    print("2. Video recording settings")
    print("3. ArUco marker settings")
    print("4. Reference marker settings")
    print("5. Visualization settings")
    print("6. Return to main menu")
    
    choice = input("Enter your choice (1-6): ")
    
    if choice == '1':
        # Camera settings
        print("\n-- Camera Settings --")
        CAMERA_INDEX = int(input(f"Camera index (current: {CAMERA_INDEX}): ") or CAMERA_INDEX)
        
        width = input(f"Camera width (current: {CAMERA_RESOLUTION[0]}, press Enter to keep): ")
        height = input(f"Camera height (current: {CAMERA_RESOLUTION[1]}, press Enter to keep): ")
        
        if width and height:
            CAMERA_RESOLUTION = (int(width), int(height))
        
        CAMERA_FPS = int(input(f"Camera FPS (current: {CAMERA_FPS}): ") or CAMERA_FPS)
        
        reverse = input(f"Reverse camera display (y/n, current: {'y' if REVERSE_CAMERA_DISPLAY else 'n'}): ").lower()
        if reverse in ['y', 'n']:
            REVERSE_CAMERA_DISPLAY = (reverse == 'y')
            
    elif choice == '2':
        # Video recording settings
        print("\n-- Video Recording Settings --")
        
        record = input(f"Record video by default (y/n, current: {'y' if RECORD_VIDEO else 'n'}): ").lower()
        if record in ['y', 'n']:
            RECORD_VIDEO = (record == 'y')
        
        codec = input(f"Video codec (current: {VIDEO_CODEC}, options: XVID, MJPG, H264): ").upper()
        if codec in ['XVID', 'MJPG', 'H264']:
            VIDEO_CODEC = codec
        
        VIDEO_FPS = int(input(f"Video FPS (current: {VIDEO_FPS}): ") or VIDEO_FPS)
            
    elif choice == '3':
        # ArUco marker settings
        print("\n-- ArUco Marker Settings --")
        
        print("ArUco Dictionary Types:")
        print("1. DICT_4X4_50    2. DICT_4X4_100   3. DICT_4X4_250   4. DICT_4X4_1000")
        print("5. DICT_5X5_50    6. DICT_5X5_100   7. DICT_5X5_250   8. DICT_5X5_1000")
        print("9. DICT_6X6_50    10. DICT_6X6_100  11. DICT_6X6_250  12. DICT_6X6_1000")
        print("13. DICT_7X7_50   14. DICT_7X7_100  15. DICT_7X7_250  16. DICT_7X7_1000")
        
        dict_choice = input(f"Select ArUco dictionary type (1-16, current: DICT_6X6_250): ")
        if dict_choice.isdigit() and 1 <= int(dict_choice) <= 16:
            dict_map = {
                '1': aruco.DICT_4X4_50, '2': aruco.DICT_4X4_100, 
                '3': aruco.DICT_4X4_250, '4': aruco.DICT_4X4_1000,
                '5': aruco.DICT_5X5_50, '6': aruco.DICT_5X5_100, 
                '7': aruco.DICT_5X5_250, '8': aruco.DICT_5X5_1000,
                '9': aruco.DICT_6X6_50, '10': aruco.DICT_6X6_100, 
                '11': aruco.DICT_6X6_250, '12': aruco.DICT_6X6_1000,
                '13': aruco.DICT_7X7_50, '14': aruco.DICT_7X7_100, 
                '15': aruco.DICT_7X7_250, '16': aruco.DICT_7X7_1000
            }
            ARUCO_DICT_TYPE = dict_map[dict_choice]
        
        marker_size = input(f"Marker size in meters (current: {MARKER_SIZE_METERS}): ")
        if marker_size:
            MARKER_SIZE_METERS = float(marker_size)
            
    elif choice == '4':
        # Reference marker settings
        print("\n-- Reference Marker Settings --")
        
        use_ref = input(f"Use reference marker (y/n, current: {'y' if USE_REFERENCE_MARKER else 'n'}): ").lower()
        if use_ref in ['y', 'n']:
            USE_REFERENCE_MARKER = (use_ref == 'y')
        
        if USE_REFERENCE_MARKER:
            ref_id = input(f"Reference marker ID (current: {REFERENCE_MARKER_ID}): ")
            if ref_id:
                REFERENCE_MARKER_ID = int(ref_id)
                
    elif choice == '5':
        # Visualization settings
        print("\n-- Visualization Settings --")
        
        traj_len = input(f"Trajectory length (current: {TRAJECTORY_LENGTH}): ")
        if traj_len:
            TRAJECTORY_LENGTH = int(traj_len)
            
    # Return to main menu for option 6 or invalid choices
    return

if __name__ == "__main__":
    # Create directories if they don't exist
    if not os.path.exists(DATA_OUTPUT_DIRECTORY):
        os.makedirs(DATA_OUTPUT_DIRECTORY)
    
    video_path = os.path.join(DATA_OUTPUT_DIRECTORY, VIDEO_DIRECTORY)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            # Generate ArUco marker
            marker_id = int(input("Enter marker ID (0-249 for DICT_6X6_250): "))
            marker_size = int(input("Enter marker size in pixels (default: 600): ") or "600")
            marker = generate_aruco_marker(marker_id, marker_size)
            
            # Display the marker
            cv2.imshow(f"ArUco Marker ID: {marker_id}", marker)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif choice == '2':
            # Generate reference marker
            marker_size = int(input("Enter marker size in pixels (default: 600): ") or "600")
            marker = generate_aruco_marker(REFERENCE_MARKER_ID, marker_size)
            
            # Display the marker
            cv2.imshow(f"Reference Marker ID: {REFERENCE_MARKER_ID}", marker)
            print(f"This marker (ID: {REFERENCE_MARKER_ID}) will be used as the reference coordinate system.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif choice == '3':
            # Generate chessboard pattern
            squares_x = int(input(f"Enter number of squares in X direction (default: {CALIBRATION_CHESSBOARD_SHAPE[0]}): ") 
                          or str(CALIBRATION_CHESSBOARD_SHAPE[0]))
            squares_y = int(input(f"Enter number of squares in Y direction (default: {CALIBRATION_CHESSBOARD_SHAPE[1]}): ")
                          or str(CALIBRATION_CHESSBOARD_SHAPE[1]))
            square_size = int(input("Enter square size in pixels (default: 100): ") or "100")
            
            chessboard = create_chessboard(squares_x, squares_y, square_size)
            
            # Display the chessboard
            cv2.imshow("Chessboard Pattern", chessboard)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif choice == '4':
            # Calibrate camera
            calibrate_camera()
            
        elif choice == '5':
            # Start buoy tracking
            use_ref = input(f"Use marker ID {REFERENCE_MARKER_ID} as reference frame? (y/n, default: {'y' if USE_REFERENCE_MARKER else 'n'}): ").lower() or ('y' if USE_REFERENCE_MARKER else 'n')
            USE_REFERENCE_MARKER = (use_ref == 'y')
            
            record = input(f"Record video? (y/n, default: {'y' if RECORD_VIDEO else 'n'}): ").lower() or ('y' if RECORD_VIDEO else 'n')
            RECORD_VIDEO = (record == 'y')
            
            reverse = input(f"Reverse camera display? (y/n, default: {'y' if REVERSE_CAMERA_DISPLAY else 'n'}): ").lower() or ('y' if REVERSE_CAMERA_DISPLAY else 'n')
            REVERSE_CAMERA_DISPLAY = (reverse == 'y')
            
            if USE_REFERENCE_MARKER:
                print(f"Using marker ID {REFERENCE_MARKER_ID} as reference coordinate system.")
            else:
                print("Using camera as reference coordinate system.")
                
            print(f"Video recording: {'ON' if RECORD_VIDEO else 'OFF'}")
            print(f"Camera display reversed: {'ON' if REVERSE_CAMERA_DISPLAY else 'OFF'}")
            
            track_buoy_with_aruco_6dof()
            
        elif choice == '6':
            # Configure settings
            configure_settings()
            
        elif choice == '7':
            # Exit
            print("Exiting program.")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")