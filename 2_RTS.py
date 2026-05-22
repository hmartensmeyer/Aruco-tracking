import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import tempfile
import scipy.linalg
from scipy.signal import medfilt
import threading
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import queue
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback

# --- Default Kalman Filter Parameters ---
DEFAULT_KF_PROCESS_NOISE_ACC = (0.5, 0.5, 0.5)
DEFAULT_KF_PROCESS_NOISE_VEL_ORI = (0.2, 0.2, 0.2)
DEFAULT_KF_MEASUREMENT_NOISE_POS = (0.0040, 0.0032, 0.0138) # X, Y, Z (m)
DEFAULT_KF_MEASUREMENT_NOISE_ORI = (0.0036, 0.0046, 0.0022) # X, Y, Z (rad)


class KalmanFilterPose6DOF_Quat:
    def __init__(self, process_noise_acc_comps, process_noise_vel_ori_comps,
                 measurement_noise_pos_comps, measurement_noise_ori_comps):
        # State vector [pos(3), vel(3), acc(3), quat(4)] -> 13 elements
        # Error state vector [d_pos(3), d_vel(3), d_acc(3), d_theta(3)] -> 12 elements
        self.x_nominal = np.zeros(13)
        self.x_nominal[12] = 1.0 # Initialize quaternion to identity (0,0,0,1)
        self.P_error = np.eye(12) * 1000.0 # Covariance of the error state
        self.F_error = np.eye(12) # State transition matrix for error state
        self.Q_error = np.zeros((12, 12)) # Process noise covariance
        # Measurement matrix H maps the error state to the measurement residual
        self.H_error = np.zeros((6, 12))
        self.H_error[0, 0] = 1; self.H_error[1, 3] = 1; self.H_error[2, 6] = 1 # Position
        self.H_error[3, 9] = 1; self.H_error[4, 10] = 1; self.H_error[5, 11] = 1 # Orientation
        
        self._noise_acc_sigma_comps = np.array(process_noise_acc_comps)
        self._noise_vel_ori_sigma_comps = np.array(process_noise_vel_ori_comps)
        r_pos_sq = np.array(measurement_noise_pos_comps)**2
        r_ori_sq = np.array(measurement_noise_ori_comps)**2
        self.R_error = np.diag(np.concatenate((r_pos_sq, r_ori_sq)))
        self.initialized = False

    def predict(self, dt):
        """Predicts the next state using a constant acceleration model."""
        if not self.initialized: return
        
        pos, vel, acc = self.x_nominal[0:3], self.x_nominal[3:6], self.x_nominal[6:9]
        
        # Predict nominal state
        self.x_nominal[0:3] = pos + vel * dt + 0.5 * acc * dt**2
        self.x_nominal[3:6] = vel + acc * dt
        # self.x_nominal[6:9] remains acc (constant acceleration model)
        # self.x_nominal[9:13] remains quat (constant orientation model for predict)

        # Predict error state covariance
        F_block_1D_pva = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
        self.F_error = scipy.linalg.block_diag(F_block_1D_pva, F_block_1D_pva, F_block_1D_pva, np.eye(3))
        
        Q_block_1D_template = np.array([[(dt**4)/4, (dt**3)/2, (dt**2)/2],
                                        [(dt**3)/2, (dt**2),   dt],
                                        [(dt**2)/2, dt,        1]])
        Q_x_pva = Q_block_1D_template * self._noise_acc_sigma_comps[0]**2
        Q_y_pva = Q_block_1D_template * self._noise_acc_sigma_comps[1]**2
        Q_z_pva = Q_block_1D_template * self._noise_acc_sigma_comps[2]**2
        q_ori_vals_sq = (self._noise_vel_ori_sigma_comps * dt)**2
        Q_ori = np.diag(q_ori_vals_sq)
        self.Q_error = scipy.linalg.block_diag(Q_x_pva, Q_y_pva, Q_z_pva, Q_ori)
        
        self.P_error = self.F_error @ self.P_error @ self.F_error.T + self.Q_error

    def update(self, z_measurement_pose):
        """Updates the state with a new measurement."""
        t_meas, q_meas = z_measurement_pose[:3], z_measurement_pose[3:7]

        if not self.initialized:
            # First measurement initializes the state
            self.x_nominal[0:3] = t_meas.copy()
            self.x_nominal[9:13] = q_meas.copy()
            norm_q_init = np.linalg.norm(self.x_nominal[9:13])
            if norm_q_init > 1e-9: self.x_nominal[9:13] /= norm_q_init
            else: self.x_nominal[9:13] = np.array([0.,0.,0.,1.])
            
            # Set initial covariance based on measurement noise
            current_P_diag = np.diag(self.P_error).copy()
            current_P_diag[0] = self.R_error[0,0] * 0.1; current_P_diag[1] = 1e2; current_P_diag[2] = 1e2; # Pos X, Vel X, Acc X
            current_P_diag[3] = self.R_error[1,1] * 0.1; current_P_diag[4] = 1e2; current_P_diag[5] = 1e2; # Pos Y, Vel Y, Acc Y
            current_P_diag[6] = self.R_error[2,2] * 0.1; current_P_diag[7] = 1e2; current_P_diag[8] = 1e2; # Pos Z, Vel Z, Acc Z
            current_P_diag[9:12] = [self.R_error[3,3] * 0.1, self.R_error[4,4] * 0.1, self.R_error[5,5] * 0.1] # Orientation error
            np.fill_diagonal(self.P_error, current_P_diag)
            
            self.initialized = True
            return True

        # Calculate measurement residual (error)
        delta_p_obs = t_meas - self.x_nominal[0:3]
        
        q_nom_curr = self.x_nominal[9:13]
        q_nom_norm = np.linalg.norm(q_nom_curr)
        q_nom_obj = R.identity() if q_nom_norm < 1e-9 else R.from_quat(q_nom_curr / q_nom_norm)
        
        q_meas_norm = np.linalg.norm(q_meas)
        q_meas_obj = R.identity() if q_meas_norm < 1e-9 else R.from_quat(q_meas / q_meas_norm)
        
        q_err_obj = q_meas_obj * q_nom_obj.inv()
        delta_theta_obs = q_err_obj.as_rotvec()
        y_error = np.concatenate((delta_p_obs, delta_theta_obs))
        
        # Kalman gain calculation
        S_error = self.H_error @ self.P_error @ self.H_error.T + self.R_error
        
        try:
            K_error = self.P_error @ self.H_error.T @ np.linalg.inv(S_error)
        except np.linalg.LinAlgError:
            K_error = self.P_error @ self.H_error.T @ np.linalg.pinv(S_error) # Use pseudo-inverse if singular

        # Update error state and covariance
        delta_x_error_hat = K_error @ y_error
        self.P_error = (np.eye(12) - K_error @ self.H_error) @ self.P_error
        
        # Inject error correction into the nominal state
        self.x_nominal[0:3] += delta_x_error_hat[[0, 3, 6]]
        self.x_nominal[3:6] += delta_x_error_hat[[1, 4, 7]]
        self.x_nominal[6:9] += delta_x_error_hat[[2, 5, 8]]
        
        delta_q_obj = R.from_rotvec(delta_x_error_hat[9:12])
        updated_q_obj = delta_q_obj * q_nom_obj
        self.x_nominal[9:13] = updated_q_obj.as_quat()
        
        # Normalize the updated quaternion
        norm_q_updated = np.linalg.norm(self.x_nominal[9:13])
        if norm_q_updated > 1e-9: self.x_nominal[9:13] /= norm_q_updated
        else: self.x_nominal[9:13] = np.array([0.,0.,0.,1.])

        return True

    def get_state(self):
        return self.x_nominal.copy()


def time_str_to_seconds(time_str):
    if pd.isna(time_str): return np.nan
    try:
        parts = time_str.split(':')
        h = int(parts[0])
        m = int(parts[1])
        s_ms = parts[2].split('.')
        s = int(s_ms[0])
        ms = int(s_ms[1]) if len(s_ms) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0
    except:
        return np.nan

def average_quaternions_weighted(quaternions, weights):
    if not quaternions: return R.identity().as_quat()
    if len(quaternions) == 1: return quaternions[0]
    normalized_quaternions = []
    valid_weights = []
    for q, w_val in zip(quaternions, weights):
        if w_val > 1e-9:
            norm_q = np.linalg.norm(q)
            if norm_q < 1e-9: continue
            q_norm = q / norm_q
            normalized_quaternions.append(q_norm)
            valid_weights.append(w_val)
    if not normalized_quaternions: return R.identity().as_quat()
    if len(normalized_quaternions) == 1: return normalized_quaternions[0]
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


class KFTunerApp:
    def __init__(self, master):
        self.master = master
        master.title("Offline Kalman Smoother Tuner")
        master.geometry("700x900")

        main_canvas = tk.Canvas(master)
        scrollbar = ttk.Scrollbar(master, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame = ttk.Frame(main_canvas, padding="10")
        canvas_window = main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def on_frame_configure(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))

        def on_canvas_configure(event):
            main_canvas.itemconfig(canvas_window, width=event.width)

        self.scrollable_frame.bind("<Configure>", on_frame_configure)
        main_canvas.bind("<Configure>", on_canvas_configure)
        
        top_level_frame = ttk.Frame(self.scrollable_frame)
        top_level_frame.pack(fill=tk.BOTH, expand=True)
        
        controls_frame = ttk.Frame(top_level_frame)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        plot_frame_container = ttk.Frame(top_level_frame)
        plot_frame_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.data_df = None
        self.timestamps_sec = None
        self.raw_positions_cam = None
        self.raw_quats_cam = None
        self.buoy_ids_in_file = []
        self.plot_html_path = None
        self.hist_plot_html_path = None
        
        self.convergence_queue = queue.Queue()
        self.tune_iteration = 0
        self.is_tuning = False

        self.pos_cols_for_filter = ['CoG_Pos_X_Rel', 'CoG_Pos_Y_Rel', 'CoG_Pos_Z_Rel']
        self.quat_cols_for_filter = ['CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel']

        ttk.Button(controls_frame, text="Load CoG CSV File", command=self.load_csv).pack(pady=5, fill=tk.X)
        
        self.buoy_id_var = tk.StringVar()
        ttk.Label(controls_frame, text="Buoy ID to Analyze:").pack(pady=(10,0), anchor='w')
        self.buoy_id_selector = ttk.Combobox(controls_frame, textvariable=self.buoy_id_var, state="readonly", width=30)
        self.buoy_id_selector.pack(pady=2, fill=tk.X)
        self.buoy_id_selector.bind("<<ComboboxSelected>>", self.on_buoy_id_change)
        
        pre_processing_frame = ttk.LabelFrame(controls_frame, text="Interpolation & Filtering Pre-processing", padding="10")
        pre_processing_frame.pack(pady=(15, 5), fill=tk.X)

        median_frame = ttk.Frame(pre_processing_frame)
        median_frame.pack(pady=2, fill=tk.X)
        ttk.Label(median_frame, text="Median Kernel Size:").pack(side=tk.LEFT)
        self.median_kernel_size_var = tk.StringVar(value="5")
        self.median_entry = ttk.Entry(median_frame, textvariable=self.median_kernel_size_var, width=5)
        self.median_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(pre_processing_frame, text="(Note: Cubic spline interpolation is always applied first to fill data gaps.)", wraplength=250, justify=tk.LEFT).pack(pady=(5,0), fill=tk.X)


        noise_measure_frame = ttk.LabelFrame(controls_frame, text="Measure Noise (R) & Plot Distributions", padding="5")
        noise_measure_frame.pack(pady=(15,5), fill=tk.X)
        time_range_frame = ttk.Frame(noise_measure_frame)
        time_range_frame.pack(fill=tk.X)
        ttk.Label(time_range_frame, text="Static Start (s):").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        self.static_start_time_var = tk.StringVar(value="0.0")
        ttk.Entry(time_range_frame, textvariable=self.static_start_time_var, width=7).grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(time_range_frame, text="End (s):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
        self.static_end_time_var = tk.StringVar(value="1.0")
        ttk.Entry(time_range_frame, textvariable=self.static_end_time_var, width=7).grid(row=0, column=3, padx=2, pady=2)
        time_range_frame.columnconfigure(1, weight=1); time_range_frame.columnconfigure(3, weight=1)
        ttk.Button(noise_measure_frame, text="Measure Noise & Plot Histograms", command=self.measure_noise_from_data).pack(pady=5, fill=tk.X)
        
        kf_params_outer_frame = ttk.LabelFrame(controls_frame, text="Kalman Smoother Parameters", padding="5")
        kf_params_outer_frame.pack(pady=(10,5), fill=tk.BOTH, expand=True)
        
        convergence_frame = ttk.LabelFrame(plot_frame_container, text="Auto-Tune Convergence", padding=5)
        convergence_frame.pack(pady=(10,5), fill=tk.BOTH, expand=True)
        
        self.fig_conv = Figure(figsize=(4, 3), dpi=100)
        self.ax_conv = self.fig_conv.add_subplot(111)
        self.ax_conv.set_title("NLL Cost vs. Iteration")
        self.ax_conv.set_xlabel("Iteration")
        self.ax_conv.set_ylabel("Cost (NLL)")
        self.ax_conv.grid(True)
        self.line_conv, = self.ax_conv.plot([], [], 'r-o', markersize=3, label='Cost')
        self.fig_conv.tight_layout()

        self.canvas_conv = FigureCanvasTkAgg(self.fig_conv, master=convergence_frame)
        self.canvas_conv.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas_conv.draw()

        autotune_controls_frame = ttk.LabelFrame(kf_params_outer_frame, text="Auto-Tune Controls", padding="5")
        autotune_controls_frame.pack(pady=(5,10), fill=tk.X, padx=5)

        tune_time_frame = ttk.Frame(autotune_controls_frame)
        tune_time_frame.pack(fill=tk.X, pady=5)

        ttk.Label(tune_time_frame, text="Tune Start (s):").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        self.tune_start_time_var = tk.StringVar(value="0.0")
        ttk.Entry(tune_time_frame, textvariable=self.tune_start_time_var, width=8).grid(row=0, column=1, padx=2, pady=2, sticky=tk.EW)

        ttk.Label(tune_time_frame, text="Tune End (s):").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
        self.tune_end_time_var = tk.StringVar(value="") 
        ttk.Entry(tune_time_frame, textvariable=self.tune_end_time_var, width=8).grid(row=0, column=3, padx=2, pady=2, sticky=tk.EW)

        tune_time_frame.columnconfigure(1, weight=1)
        tune_time_frame.columnconfigure(3, weight=1)

        self.autotune_button = ttk.Button(autotune_controls_frame, text="⚡ Auto-Tune Process Noise (Q)", command=self.run_auto_tune_q)
        self.autotune_button.pack(pady=(5,0), fill=tk.X)
        
        kf_canvas = tk.Canvas(kf_params_outer_frame, height=270)
        kf_scrollbar = ttk.Scrollbar(kf_params_outer_frame, orient="vertical", command=kf_canvas.yview)
        kf_scrollable_frame = ttk.Frame(kf_canvas)
        kf_scrollable_frame.bind("<Configure>", lambda e: kf_canvas.configure(scrollregion=kf_canvas.bbox("all")))
        kf_canvas_window = kf_canvas.create_window((0, 0), window=kf_scrollable_frame, anchor="nw")
        kf_canvas.configure(yscrollcommand=kf_scrollbar.set)
        kf_canvas.pack(side="left", fill="both", expand=True)
        kf_scrollbar.pack(side="right", fill="y")
        def frame_width(event): kf_canvas.itemconfig(kf_canvas_window, width=event.width)
        kf_canvas.bind("<Configure>", frame_width)
        
        self.kf_params = {}
        axes = ['X', 'Y', 'Z']
        param_groups = {
            "Q - Proc Noise Acc (m/s^2)": ("proc_noise_acc", DEFAULT_KF_PROCESS_NOISE_ACC, (0.001, 5.0)),
            "Q - Proc Noise Vel Ori (rad/s)": ("proc_noise_vel_ori", DEFAULT_KF_PROCESS_NOISE_VEL_ORI, (0.001, 2.0)),
            "R - Meas Noise Pos (m)": ("meas_noise_pos", DEFAULT_KF_MEASUREMENT_NOISE_POS, (0.0001, 0.5)),
            "R - Meas Noise Ori (rad)": ("meas_noise_ori", DEFAULT_KF_MEASUREMENT_NOISE_ORI, (0.0001, 0.5)),
        }
        for group_label, (base_key, default_vals, (min_val, max_val)) in param_groups.items():
            ttk.Label(kf_scrollable_frame, text=f"{group_label}:").pack(pady=(8,2), anchor='w', padx=5)
            for i, axis_label in enumerate(axes):
                key = f"{base_key}_{axis_label.lower()}"
                self.kf_params[key] = tk.DoubleVar(value=default_vals[i])
                axis_frame = ttk.Frame(kf_scrollable_frame)
                axis_frame.pack(fill=tk.X, padx=15, pady=1)
                ttk.Label(axis_frame, text=f"  {axis_label}:").pack(side=tk.LEFT, padx=(0,5))
                ttk.Entry(axis_frame, textvariable=self.kf_params[key], width=8).pack(side=tk.RIGHT, padx=(5,0))
                ttk.Scale(axis_frame, from_=min_val, to=max_val, variable=self.kf_params[key], orient=tk.HORIZONTAL).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        ttk.Button(controls_frame, text="Process Data & Generate Plot", command=self.run_rts_smoother).pack(pady=10, fill=tk.X)
        
        batch_frame = ttk.LabelFrame(controls_frame, text="Batch Workflow", padding=5)
        batch_frame.pack(pady=(15, 5), fill=tk.X)
        self.batch_button = ttk.Button(batch_frame, text="Tune & Process All Buoys...", command=self.batch_process_all_buoys)
        self.batch_button.pack(pady=5, fill=tk.X)
        self.progress_bar = ttk.Progressbar(batch_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(pady=(0,5), fill=tk.X, expand=True)

        self.status_var = tk.StringVar(value="Status: Load a CSV file.")
        ttk.Label(self.scrollable_frame, textvariable=self.status_var, wraplength=380).pack(side=tk.BOTTOM, pady=5, fill=tk.X, anchor='sw')
        
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.is_tuning = False
        if self.plot_html_path and os.path.exists(self.plot_html_path):
            try: os.remove(self.plot_html_path)
            except Exception as e: print(f"Error removing temp plot file: {e}")
        if self.hist_plot_html_path and os.path.exists(self.hist_plot_html_path):
            try: os.remove(self.hist_plot_html_path)
            except Exception as e: print(f"Error removing temp histogram plot file: {e}")
        self.master.destroy()

    def load_csv(self):
        filepath = filedialog.askopenfilename(title="Select Buoy CoG CSV File", filetypes=(("CSV files", "*.csv"),))
        if not filepath: return
        try:
            self.status_var.set("Status: Loading CSV...")
            self.master.update_idletasks()
            df = pd.read_csv(filepath, na_values=["NaN", " NaN", "NaN ", "", " NA", "NA "])
            required_cols = ['Date', 'Time', 'Buoy_ID'] + self.pos_cols_for_filter + self.quat_cols_for_filter
            if not all(col in df.columns for col in required_cols):
                tk.messagebox.showerror("CSV Error", f"Missing required columns: {', '.join(col for col in required_cols if col not in df.columns)}")
                self.status_var.set("Status: CSV loading error."); return
            df['Timestamp_sec'] = df['Time'].apply(time_str_to_seconds)
            df = df.sort_values(by=['Date', 'Timestamp_sec']).reset_index(drop=True)
            if df['Timestamp_sec'].notna().any():
                 first_valid_ts = df['Timestamp_sec'][df['Timestamp_sec'].notna()].iloc[0]
                 df['Time_Rel_Sec'] = df['Timestamp_sec'] - first_valid_ts
            else: df['Time_Rel_Sec'] = pd.Series(dtype=float)
            self.data_df = df
            self.buoy_ids_in_file = sorted(self.data_df['Buoy_ID'].dropna().unique().astype(int).tolist())
            self.buoy_id_selector['values'] = self.buoy_ids_in_file
            if self.buoy_ids_in_file:
                self.buoy_id_var.set(str(self.buoy_ids_in_file[0]))
                self.on_buoy_id_change(None)
            else: self.buoy_id_var.set("")
            self.status_var.set(f"Status: Loaded {os.path.basename(filepath)}. Select Buoy ID.")
        except Exception as e:
            tk.messagebox.showerror("Error Loading CSV", f"An error occurred: {e}")
            self.status_var.set(f"Status: Error loading CSV: {e}"); self.data_df = None

    def on_buoy_id_change(self, event):
        if not self._prepare_buoy_data():
            self.tune_start_time_var.set("0.0")
            self.tune_end_time_var.set("")
            self.status_var.set(f"Status: Failed to prepare data for Buoy ID {self.buoy_id_var.get()}.")
            return
        
        if self.timestamps_sec is not None and np.any(~np.isnan(self.timestamps_sec)):
            min_time = np.nanmin(self.timestamps_sec)
            max_time = np.nanmax(self.timestamps_sec)
            self.tune_start_time_var.set(f"{min_time:.2f}")
            self.tune_end_time_var.set(f"{max_time:.2f}")
        else:
            self.tune_start_time_var.set("0.0")
            self.tune_end_time_var.set("")

        self.status_var.set(f"Status: Buoy ID {self.buoy_id_var.get()} selected. Adjust params or measure noise.")

    def _prepare_buoy_data(self):
        if self.data_df is None or not self.buoy_id_var.get():
            self.timestamps_sec, self.raw_positions_cam, self.raw_quats_cam = None, None, None
            return False
        try: selected_buoy_id = int(self.buoy_id_var.get())
        except ValueError: self.status_var.set("Status: Invalid Buoy ID."); return False
        buoy_df = self.data_df[self.data_df['Buoy_ID'] == selected_buoy_id].copy()
        if buoy_df.empty:
            self.status_var.set(f"Status: No data for Buoy ID {selected_buoy_id}.")
            self.timestamps_sec, self.raw_positions_cam, self.raw_quats_cam = None, None, None
            return False
        self.timestamps_sec = buoy_df['Time_Rel_Sec'].values
        self.raw_positions_cam = buoy_df[self.pos_cols_for_filter].values
        self.raw_quats_cam = buoy_df[self.quat_cols_for_filter].values
        return True

    def measure_noise_from_data(self):
        if not self._prepare_buoy_data() or self.timestamps_sec is None:
            messagebox.showerror("Error", "Please load data and select a Buoy ID first."); return
        try:
            start_t, end_t = float(self.static_start_time_var.get()), float(self.static_end_time_var.get())
            if start_t >= end_t: messagebox.showerror("Input Error", "Start time must be less than end time."); return
        except ValueError: messagebox.showerror("Input Error", "Invalid start/end time."); return

        pos_devs_hist, quats_hist = None, None
        pos_std_msg = np.array([self.kf_params["meas_noise_pos_x"].get(), self.kf_params["meas_noise_pos_y"].get(), self.kf_params["meas_noise_pos_z"].get()])
        ori_std_msg = np.array([self.kf_params["meas_noise_ori_x"].get(), self.kf_params["meas_noise_ori_y"].get(), self.kf_params["meas_noise_ori_z"].get()])
        
        mask = (self.timestamps_sec >= start_t) & (self.timestamps_sec <= end_t)
        static_pos = self.raw_positions_cam[mask]
        if len(static_pos) >= 2:
            pos_std_calc = np.nanstd(static_pos, axis=0)
            if len(pos_std_calc) == 3:
                vals = [pos_std_calc[i] if not np.isnan(pos_std_calc[i]) else getattr(self.kf_params, f"meas_noise_pos_{'xyz'[i]}").get() for i in range(3)]
                for i, k in enumerate(["x","y","z"]): self.kf_params[f"meas_noise_pos_{k}"].set(round(vals[i], 4))
                pos_std_msg = np.array(vals)
                pos_devs_hist = static_pos - np.nan_to_num(np.nanmean(static_pos, axis=0))
            else: 
                avg_std = np.nanmean(pos_std_calc) if pos_std_calc.size > 0 else DEFAULT_KF_MEASUREMENT_NOISE_POS[0]
                avg_std = avg_std if not np.isnan(avg_std) else DEFAULT_KF_MEASUREMENT_NOISE_POS[0]
                for k in ["x","y","z"]: self.kf_params[f"meas_noise_pos_{k}"].set(round(avg_std, 4))
                pos_std_msg = np.full(3, avg_std)
        else: messagebox.showwarning("Warning", "Not enough position data for noise estimate.")

        static_quat = self.raw_quats_cam[mask]; valid_static_quat = static_quat[~np.isnan(static_quat).any(axis=1)]
        if len(valid_static_quat) >= 2:
            norm_quats = [q/np.linalg.norm(q) if np.linalg.norm(q) > 1e-9 else np.array([0,0,0,1]) for q in valid_static_quat]
            if norm_quats:
                quats_hist = np.array(norm_quats)
                mean_q = average_quaternions_weighted(norm_quats, [1.0]*len(norm_quats))
                if not np.isnan(mean_q).any():
                    mean_q_obj = R.from_quat(mean_q)
                    err_vecs = [ (R.from_quat(q_m) * mean_q_obj.inv()).as_rotvec() for q_m in norm_quats ]
                    if err_vecs:
                        ori_std_calc = np.nanstd(np.array(err_vecs), axis=0)
                        if len(ori_std_calc) == 3:
                            vals = [ori_std_calc[i] if not np.isnan(ori_std_calc[i]) else getattr(self.kf_params, f"meas_noise_ori_{'xyz'[i]}").get() for i in range(3)]
                            ori_std_msg = np.array(vals)
                        else:
                            avg_std = np.nanmean(ori_std_calc) if ori_std_calc.size > 0 else DEFAULT_KF_MEASUREMENT_NOISE_ORI[0]
                            avg_std = avg_std if not np.isnan(avg_std) else DEFAULT_KF_MEASUREMENT_NOISE_ORI[0]
                            ori_std_msg = np.full(3, avg_std)
                        for i, k in enumerate(["x","y","z"]): self.kf_params[f"meas_noise_ori_{k}"].set(round(ori_std_msg[i], 4))
        else: messagebox.showwarning("Warning", "Not enough quaternion data for noise estimate.")
        
        status_msg = (f"Status: Measured R sigmas:\nPos: {pos_std_msg[0]:.4f}, {pos_std_msg[1]:.4f}, {pos_std_msg[2]:.4f} m\n"
                      f"Ori ErrVec: {ori_std_msg[0]:.4f}, {ori_std_msg[1]:.4f}, {ori_std_msg[2]:.4f} rad")
        self.status_var.set(status_msg)
        messagebox.showinfo("Noise Measured", f"R sigmas updated:\nPos: {pos_std_msg[0]:.4f}, {pos_std_msg[1]:.4f}, {pos_std_msg[2]:.4f} m\n"
                                          f"Ori ErrVec: {ori_std_msg[0]:.4f}, {ori_std_msg[1]:.4f}, {ori_std_msg[2]:.4f} rad")
        if pos_devs_hist is not None or quats_hist is not None:
            self._plot_noise_histograms(pos_devs_hist, quats_hist, start_t, end_t)
        else: self.status_var.set(self.status_var.get() + "\nNo new data for histograms.")

    def _plot_noise_histograms(self, pos_deviations, static_quaternion_components_for_hist, start_t, end_t, buoy_id_override=None, return_fig=False):
        buoy_id = buoy_id_override if buoy_id_override is not None else self.buoy_id_var.get()
        fig_title = f"Noise Measurement Distributions - Buoy ID: {buoy_id} (Static: {start_t:.2f}s - {end_t:.2f}s)"
        has_pos_data = pos_deviations is not None and pos_deviations.shape[0] > 0
        has_ori_data = static_quaternion_components_for_hist is not None and len(static_quaternion_components_for_hist) > 0
        
        if not has_pos_data and not has_ori_data:
             if not return_fig: messagebox.showinfo("Histogram Info", "No sufficient data to plot noise histograms.")
             return None

        num_rows = 0; specs = []; flat_subplot_titles = []
        if has_pos_data:
            num_rows += 1; specs.append([{}, {}, {}])
            pos_labels = ['X', 'Y', 'Z']; temp_pos_titles = []
            for i in range(3):
                data_col = pos_deviations[:, i][~np.isnan(pos_deviations[:, i])]
                title_text = f"Pos {pos_labels[i]} Dev (N={len(data_col)})"
                if len(data_col) > 0: title_text += f"<br>σ: {np.std(data_col):.4f}m"
                else: title_text += "<br>σ: N/A"
                temp_pos_titles.append(title_text)
            flat_subplot_titles.extend(temp_pos_titles)
        if has_ori_data:
            num_rows += 2; specs.append([{}, {}, None]); specs.append([{}, {}, None])
            static_quats_np = np.array(static_quaternion_components_for_hist)
            q_labels_hist = ['qx', 'qy', 'qz', 'qw']; ori_titles_for_grid = [""]*4
            for q_idx in range(4):
                data_col = static_quats_np[:, q_idx][~np.isnan(static_quats_np[:, q_idx])]
                title_text = f"{q_labels_hist[q_idx]} Comp. (N={len(data_col)})"
                if len(data_col) > 0: title_text += f"<br>σ: {np.std(data_col):.4f}"
                else: title_text += "<br>σ: N/A"
                ori_titles_for_grid[q_idx] = title_text
            flat_subplot_titles.extend([ori_titles_for_grid[0], ori_titles_for_grid[1], None, ori_titles_for_grid[2], ori_titles_for_grid[3], None])

        fig_hist = make_subplots(rows=num_rows, cols=3, specs=specs, subplot_titles=tuple(flat_subplot_titles) if flat_subplot_titles else None, vertical_spacing=0.2 if num_rows > 1 else 0.1, horizontal_spacing=0.1)
        current_fig_row = 1
        if has_pos_data:
            pos_labels = ['X', 'Y', 'Z']
            for i in range(3):
                data_col = pos_deviations[:, i][~np.isnan(pos_deviations[:, i])]
                if len(data_col) > 0:
                    fig_hist.add_trace(go.Histogram(x=data_col, name=f'Pos {pos_labels[i]} Dev.', showlegend=False), row=current_fig_row, col=i+1)
                fig_hist.update_xaxes(title_text="Deviation (m)", row=current_fig_row, col=i+1)
            current_fig_row += 1
        if has_ori_data:
            q_labels = ['qx', 'qy', 'qz', 'qw']; static_quats_np = np.array(static_quaternion_components_for_hist)
            for i_row in range(2):
                for i_col in range(2):
                    q_idx = i_row * 2 + i_col
                    if q_idx >= 4: break
                    data_col = static_quats_np[:, q_idx][~np.isnan(static_quats_np[:, q_idx])]
                    if len(data_col) > 0:
                        fig_hist.add_trace(go.Histogram(x=data_col, name=f'{q_labels[q_idx]}', showlegend=False), row=current_fig_row + i_row, col=i_col + 1)
                    fig_hist.update_xaxes(title_text="Quat. Comp.", row=current_fig_row + i_row, col=i_col + 1)
        
        fig_hist.update_layout(title_text=fig_title, height=max(300, 220 * num_rows + 100), showlegend=False, margin=dict(t=120), title_x=0.5)

        if return_fig:
            return fig_hist
            
        if self.hist_plot_html_path and os.path.exists(self.hist_plot_html_path):
            try: os.remove(self.hist_plot_html_path)
            except Exception as e: print(f"Warning: Could not remove old histogram file: {e}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix="kf_hist_plot_") as tmp_file:
            self.hist_plot_html_path = tmp_file.name
        try:
            fig_hist.write_html(self.hist_plot_html_path)
            webbrowser.open_new_tab(f'file://{os.path.realpath(self.hist_plot_html_path)}')
            self.status_var.set(self.status_var.get() + "\nNoise component histograms plotted.")
        except Exception as e: messagebox.showerror("Plotting Error", f"Could not generate or open histogram plot: {e}")

    def _execute_smoother_logic(self, timestamps, positions, quaternions, kf_params_dict):
        """
        Core RTS smoother logic. It assumes input data is DENSE (no NaNs)
        because pre-processing has already filled the gaps.
        """
        pna_comps = kf_params_dict["proc_noise_acc"]
        pnvo_comps = kf_params_dict["proc_noise_vel_ori"]
        mnp_comps = kf_params_dict["meas_noise_pos"]
        mno_comps = kf_params_dict["meas_noise_ori"]

        kf = KalmanFilterPose6DOF_Quat(pna_comps, pnvo_comps, mnp_comps, mno_comps)

        dts = np.diff(timestamps, prepend=timestamps[0])
        dts[dts <= 1e-6] = (1/30.0)
        
        num_steps = len(timestamps)
        x_filtered_hist = [np.zeros(13)] * num_steps
        P_filtered_hist = [np.eye(12)] * num_steps
        x_predicted_hist = [np.zeros(13)] * num_steps
        P_predicted_hist = [np.eye(12)] * num_steps
        F_hist = [np.eye(12)] * num_steps

        # Forward Pass (Kalman Filter)
        for i in range(num_steps):
            dt = dts[i]
            meas_pose = np.concatenate((positions[i], quaternions[i]))
            
            # Predict step
            kf.predict(dt)
            x_predicted_hist[i], P_predicted_hist[i], F_hist[i] = kf.x_nominal.copy(), kf.P_error.copy(), kf.F_error.copy()

            # Update step (always runs as data is dense)
            kf.update(meas_pose)
            x_filtered_hist[i], P_filtered_hist[i] = kf.x_nominal.copy(), kf.P_error.copy()

        # Backward Pass (RTS Smoother)
        x_smoothed_hist, P_smoothed_hist = x_filtered_hist[:], P_filtered_hist[:]
        for k in range(num_steps - 2, -1, -1):
            P_k_updated, P_k1_predicted = P_filtered_hist[k], P_predicted_hist[k+1]
            try: 
                C_k = P_k_updated @ F_hist[k+1].T @ np.linalg.pinv(P_k1_predicted)
            except np.linalg.LinAlgError: 
                continue # Skip if prediction covariance is singular
            
            # Calculate the difference between smoothed and predicted states at k+1
            delta_pva_k1 = x_smoothed_hist[k+1][:9] - x_predicted_hist[k+1][:9]
            q_k1_smoothed_obj = R.from_quat(x_smoothed_hist[k+1][9:13])
            q_k1_pred_obj = R.from_quat(x_predicted_hist[k+1][9:13])
            delta_theta_k1 = (q_k1_smoothed_obj * q_k1_pred_obj.inv()).as_rotvec()
            
            delta_x_k1_error = np.zeros(12)
            delta_x_k1_error[[0,3,6,1,4,7,2,5,8]] = delta_pva_k1
            delta_x_k1_error[9:12] = delta_theta_k1
            
            # Update the state and covariance at step k
            delta_x_k_smoothed = C_k @ delta_x_k1_error
            P_smoothed_hist[k] = P_k_updated - C_k @ (P_k1_predicted - P_smoothed_hist[k+1]) @ C_k.T
            
            x_k_filtered = x_filtered_hist[k].copy()
            x_k_filtered[0:3] += delta_x_k_smoothed[[0, 3, 6]]
            x_k_filtered[3:6] += delta_x_k_smoothed[[1, 4, 7]]
            x_k_filtered[6:9] += delta_x_k_smoothed[[2, 5, 8]]
            
            q_k_filt_obj = R.from_quat(x_filtered_hist[k][9:13])
            updated_q_obj = R.from_rotvec(delta_x_k_smoothed[9:12]) * q_k_filt_obj
            x_k_filtered[9:13] = updated_q_obj.as_quat()
            x_smoothed_hist[k] = x_k_filtered
        
        all_smoothed_states_np = np.array(x_smoothed_hist)
        return timestamps, all_smoothed_states_np

    def run_rts_smoother(self):
        if not self._prepare_buoy_data() or self.timestamps_sec is None:
            self.status_var.set("Status: Load data and select Buoy ID first."); return
        
        # Store original data with NaNs for plotting comparison
        original_timestamps = self.timestamps_sec.copy()
        original_positions = self.raw_positions_cam.copy()
        original_quaternions = self.raw_quats_cam.copy()

        self.status_var.set("Status: Applying pre-processing (interpolation & filtering)..."); self.master.update_idletasks()
        processed_data = self._run_preprocessing()
        if processed_data[0] is None:
            self.status_var.set("Status: Pre-processing failed. Cannot run smoother."); return
        timestamps, positions, quaternions = processed_data
        
        self.status_var.set("Status: Running RTS smoother..."); self.master.update_idletasks()
        
        current_kf_params = {
            "proc_noise_acc": [self.kf_params[f"proc_noise_acc_{k}"].get() for k in "xyz"],
            "proc_noise_vel_ori": [self.kf_params[f"proc_noise_vel_ori_{k}"].get() for k in "xyz"],
            "meas_noise_pos": [self.kf_params[f"meas_noise_pos_{k}"].get() for k in "xyz"],
            "meas_noise_ori": [self.kf_params[f"meas_noise_ori_{k}"].get() for k in "xyz"]
        }
        
        smoothed_timestamps, smoothed_states = self._execute_smoother_logic(timestamps, positions, quaternions, current_kf_params)
        
        smooth_pos, smooth_vel = smoothed_states[:, 0:3], smoothed_states[:, 3:6]
        smooth_acc, smooth_quat = smoothed_states[:, 6:9], smoothed_states[:, 9:13]

        missing_indices = np.where(np.isnan(original_positions).any(axis=1))[0]

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=(f"Position - Buoy ID: {self.buoy_id_var.get()}", "Velocity", "Acceleration", "Orientation (Quat)"),
                            vertical_spacing=0.07)
        labels_pva, colors_plotly = ['X', 'Y', 'Z'], ['red', 'green', 'blue', 'purple']
        
        # Plot Position
        for i in range(3):
            fig.add_trace(go.Scatter(x=original_timestamps, y=original_positions[:, i], mode='markers', marker=dict(size=3, opacity=0.4), name=f'Raw Pos {labels_pva[i]}', line=dict(color=colors_plotly[i]), legendgroup='pos', legendgrouptitle_text='Position'), row=1, col=1)
            fig.add_trace(go.Scatter(x=smoothed_timestamps, y=smooth_pos[:, i], mode='lines', name=f'Smoothed Pos {labels_pva[i]}', line=dict(color=colors_plotly[i], width=2), legendgroup='pos'), row=1, col=1)
        
        if len(missing_indices) > 0:
            missing_ts = smoothed_timestamps[missing_indices]
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='circle-open', color='orange', size=8, line=dict(width=2)), name='Smoothed Gap', legendgroup='pos'), row=1, col=1)
            for i in range(3):
                fig.add_trace(go.Scatter(x=missing_ts, y=smooth_pos[missing_indices, i], mode='markers', marker=dict(symbol='circle-open', color='orange', size=8, line=dict(width=2)), showlegend=False, legendgroup='pos'), row=1, col=1)
        
        # Plot Velocity, Acceleration, and Orientation
        for i in range(3): fig.add_trace(go.Scatter(x=smoothed_timestamps, y=smooth_vel[:, i], mode='lines', name=f'Smoothed Vel {labels_pva[i]}', line=dict(color=colors_plotly[i], width=1.5), legendgroup='vel', legendgrouptitle_text='Velocity'), row=2, col=1)
        for i in range(3): fig.add_trace(go.Scatter(x=smoothed_timestamps, y=smooth_acc[:, i], mode='lines', name=f'Smoothed Acc {labels_pva[i]}', line=dict(color=colors_plotly[i], width=1.5), legendgroup='acc', legendgrouptitle_text='Acceleration'), row=3, col=1)
        labels_quat = ['qx', 'qy', 'qz', 'qw']
        for i in range(4):
            fig.add_trace(go.Scatter(x=original_timestamps, y=original_quaternions[:, i], mode='markers', marker=dict(size=3, opacity=0.4), name=f'Raw {labels_quat[i]}', line=dict(color=colors_plotly[i]), legendgroup='ori', legendgrouptitle_text='Orientation'), row=4, col=1)
            fig.add_trace(go.Scatter(x=smoothed_timestamps, y=smooth_quat[:, i], mode='lines', name=f'Smoothed {labels_quat[i]}', line=dict(color=colors_plotly[i], width=2), legendgroup='ori'), row=4, col=1)

        self.finalize_and_show_plot(fig)

    def _run_preprocessing(self, buoy_df=None):
        """
        Pre-processes buoy data by first filling gaps with spline interpolation,
        then applying a median filter. This ensures the data sent to the
        smoother is dense and clean.
        """
        if buoy_df is None:
            buoy_id = int(self.buoy_id_var.get())
            buoy_df = self.data_df[self.data_df['Buoy_ID'] == buoy_id].copy()
        
        buoy_df = buoy_df.set_index('Time_Rel_Sec').sort_index()
        interp_cols = self.pos_cols_for_filter + self.quat_cols_for_filter
        
        buoy_df[interp_cols] = buoy_df[interp_cols].interpolate(method='spline', order=3, limit_direction='both')
        
        buoy_df.dropna(subset=interp_cols, inplace=True)
        if buoy_df.empty:
            # Check if this is being called from interactive mode to show message
            is_interactive = 'buoy_df' not in locals() or buoy_df is None
            if is_interactive:
                messagebox.showerror("Error", "No valid data remains after interpolation.")
            return None, None, None
        
        processed_timestamps = buoy_df.index.to_numpy()
        processed_pos = buoy_df[self.pos_cols_for_filter].to_numpy()
        processed_quat = buoy_df[self.quat_cols_for_filter].to_numpy()
        
        try: kernel_size = int(self.median_kernel_size_var.get())
        except ValueError:
            is_interactive = 'buoy_df' not in locals() or buoy_df is None
            if is_interactive:
                messagebox.showerror("Input Error", "Median kernel size must be an integer.")
            return None, None, None
        
        if kernel_size % 2 == 0 or kernel_size < 1:
            is_interactive = 'buoy_df' not in locals() or buoy_df is None
            if is_interactive:
                messagebox.showerror("Input Error", "Median kernel size must be a positive odd integer.")
            return None, None, None
        
        if kernel_size > 1:
            for i in range(3): processed_pos[:, i] = medfilt(processed_pos[:, i], kernel_size=kernel_size)
            for i in range(4): processed_quat[:, i] = medfilt(processed_quat[:, i], kernel_size=kernel_size)

        norms = np.linalg.norm(processed_quat, axis=1)
        processed_quat[norms > 1e-9] = processed_quat[norms > 1e-9] / norms[norms > 1e-9, np.newaxis]
        
        return processed_timestamps, processed_pos, processed_quat

    def finalize_and_show_plot(self, fig):
        fig.update_layout(height=1000, hovermode='x unified', legend_tracegroupgap=150)
        fig.update_yaxes(title_text="Position (m)", row=1, col=1); fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
        fig.update_yaxes(title_text="Accel (m/s^2)", row=3, col=1); fig.update_yaxes(title_text="Quat. Comp.", row=4, col=1, range=[-1.05, 1.05])
        fig.update_xaxes(title_text="Time (s)", row=4, col=1)
        if self.plot_html_path and os.path.exists(self.plot_html_path):
            try: os.remove(self.plot_html_path)
            except Exception as e: print(f"Warning: Could not remove old plot file: {e}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix="kf_plot_") as tmp_file:
            self.plot_html_path = tmp_file.name
        try:
            fig.write_html(self.plot_html_path)
            webbrowser.open_new_tab(f'file://{os.path.realpath(self.plot_html_path)}')
            self.status_var.set("Status: Plot updated.")
        except Exception as e:
            messagebox.showerror("Plotting Error", f"Could not generate/open plot: {e}")
            self.status_var.set(f"Status: Error plotting: {e}")

    def _check_queue_and_update_plot(self):
        if not self.is_tuning:
            return
        try:
            while not self.convergence_queue.empty():
                iteration, cost = self.convergence_queue.get_nowait()
                x_data = np.append(self.line_conv.get_xdata(), iteration)
                y_data = np.append(self.line_conv.get_ydata(), cost)
                self.line_conv.set_data(x_data, y_data)
            self.ax_conv.relim()
            self.ax_conv.autoscale_view(True, True, True)
            self.fig_conv.tight_layout()
            self.canvas_conv.draw()
        except queue.Empty:
            pass
        self.master.after(200, self._check_queue_and_update_plot)

    def run_auto_tune_q(self):
        if not self._prepare_buoy_data() or self.timestamps_sec is None:
            messagebox.showerror("Error", "Please load data and select a Buoy ID first.")
            return
        try:
            start_t = float(self.tune_start_time_var.get())
            end_t_str = self.tune_end_time_var.get()
            end_t = float(end_t_str) if end_t_str.strip() else np.inf
        except ValueError:
            messagebox.showerror("Input Error", "Auto-tune start/end time must be numeric."); return
        if start_t >= end_t:
            messagebox.showerror("Input Error", "Auto-tune start time must be less than end time."); return

        processed_data = self._run_preprocessing()
        if processed_data[0] is None:
            self.status_var.set("Status: Pre-processing failed. Cannot auto-tune."); return
        
        timestamps, positions, quaternions = processed_data
        mask = (timestamps >= start_t) & (timestamps <= end_t)
        
        timestamps_sliced, positions_sliced, quaternions_sliced = timestamps[mask], positions[mask], quaternions[mask]
        
        if len(timestamps_sliced) < 20: 
            messagebox.showwarning("Data Warning", f"Found only {len(timestamps_sliced)} data points in the selected time range [{start_t:.2f}s, {end_t if np.isfinite(end_t) else 'end'}s].\n\nAuto-tuning requires more data to be effective.")
            return
        
        print(f"Auto-tuning on {len(timestamps_sliced)} data points from t={timestamps_sliced[0]:.2f}s to t={timestamps_sliced[-1]:.2f}s.")

        self.line_conv.set_data([], [])
        self.canvas_conv.draw()
        
        while not self.convergence_queue.empty():
            try: self.convergence_queue.get_nowait()
            except queue.Empty: break
            
        self.tune_iteration = 0
        self.is_tuning = True
        self.status_var.set(f"Status: Starting Auto-Tune for Q on {len(timestamps_sliced)} points... See convergence plot.")
        self.master.update_idletasks()
        self.autotune_button.config(state="disabled")
        
        self._check_queue_and_update_plot()
        thread = threading.Thread(target=self._perform_optimization, args=(timestamps_sliced, positions_sliced, quaternions_sliced))
        thread.daemon = True
        thread.start()

    def _perform_optimization(self, timestamps, positions, quaternions):
        try:
            initial_q_acc = [self.kf_params[f"proc_noise_acc_{k}"].get() for k in "xyz"]
            initial_q_ori = [self.kf_params[f"proc_noise_vel_ori_{k}"].get() for k in "xyz"]
            initial_guess = np.array(initial_q_acc + initial_q_ori)
            bounds = [(1e-3, 1.0)] * 3 + [(1e-3, 1.0)] * 3
            result = minimize(
                fun=self._cost_function_nll,
                x0=initial_guess,
                args=(timestamps, positions, quaternions),
                method='L-BFGS-B',
                options={'ftol': 1e-10, 'maxiter': 1000}
            )
            def update_gui_with_results():
                if result.success:
                    best_params = result.x
                    for i, k in enumerate("xyz"): self.kf_params[f"proc_noise_acc_{k}"].set(round(best_params[i], 5))
                    for i, k in enumerate("xyz"): self.kf_params[f"proc_noise_vel_ori_{k}"].set(round(best_params[i+3], 5))
                    final_msg = f"Status: Auto-Tune complete! Final cost: {result.fun:.4f}"
                    self.status_var.set(final_msg)
                    messagebox.showinfo("Success", f"{final_msg}\nGUI values have been updated.")
                else:
                    self.status_var.set(f"Status: Auto-Tune failed or was stopped. {result.message}")
                    messagebox.showwarning("Optimization Failed", f"The optimization did not converge: {result.message}")
            
            self.master.after(0, update_gui_with_results)

        except Exception as e:
            def show_error():
                self.status_var.set(f"Status: Error during auto-tune: {e}")
                messagebox.showerror("Auto-Tune Error", f"An error occurred during optimization: {e}")
            self.master.after(0, show_error)
        finally:
            self.is_tuning = False
            self.master.after(0, lambda: self.autotune_button.config(state="normal"))

    def _cost_function_nll(self, q_params, timestamps, positions, quaternions):
        pna_comps = q_params[0:3]
        pnvo_comps = q_params[3:6]
        pna_comps = np.maximum(pna_comps, 1e-9)
        pnvo_comps = np.maximum(pnvo_comps, 1e-9)
        mnp_comps = [self.kf_params[f"meas_noise_pos_{k}"].get() for k in "xyz"]
        mno_comps = [self.kf_params[f"meas_noise_ori_{k}"].get() for k in "xyz"]
        
        kf = KalmanFilterPose6DOF_Quat(pna_comps, pnvo_comps, mnp_comps, mno_comps)

        dts = np.diff(timestamps)
        dts = np.insert(dts, 0, dts[0] if len(dts) > 0 else 1/30.0)
        dts[dts <= 1e-6] = (1/30.0)

        total_nll = 0.0
        n_samples = 0

        for i in range(len(timestamps)):
            dt = dts[i]
            if kf.initialized: kf.predict(dt)
            t_meas, q_meas = positions[i], quaternions[i]
            
            delta_p_obs = t_meas - kf.x_nominal[0:3]
            q_nom_obj = R.from_quat(kf.x_nominal[9:13] / np.linalg.norm(kf.x_nominal[9:13]))
            q_meas_obj = R.from_quat(q_meas / np.linalg.norm(q_meas))
            delta_theta_obs = (q_meas_obj * q_nom_obj.inv()).as_rotvec()
            y_error = np.concatenate((delta_p_obs, delta_theta_obs))
            
            S_error = kf.H_error @ kf.P_error @ kf.H_error.T + kf.R_error

            if kf.initialized:
                try:
                    sign, logdet_S = np.linalg.slogdet(S_error)
                    if sign < 1: return 1e12
                    mahalanobis_dist_sq = y_error.T @ np.linalg.solve(S_error, y_error)
                    nll_step = 0.5 * (logdet_S + mahalanobis_dist_sq)
                    if np.isfinite(nll_step):
                        total_nll += nll_step; n_samples += 1
                    else: return 1e12
                except np.linalg.LinAlgError: return 1e12
            kf.update(np.concatenate((t_meas, q_meas)))

        if n_samples < 10: return 1e12
        cost = total_nll / n_samples
        if self.is_tuning:
            self.convergence_queue.put((self.tune_iteration, cost))
        self.tune_iteration += 1
        return cost
        
    def _generate_smoother_plot_fig(self, buoy_id, original_df, smoothed_df):
        """Generates the main smoother plot figure for a given buoy."""
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=(f"Position - Buoy ID: {buoy_id}", "Velocity", "Acceleration", "Orientation (Quat)"),
                            vertical_spacing=0.07)
        
        original_timestamps = original_df['Time_Rel_Sec'].values
        original_positions = original_df[self.pos_cols_for_filter].values
        original_quaternions = original_df[self.quat_cols_for_filter].values

        # To get all states for plotting, we re-run the smoother logic.
        kf_params_dict = {
            "proc_noise_acc": [self.kf_params[f"proc_noise_acc_{k}"].get() for k in "xyz"],
            "proc_noise_vel_ori": [self.kf_params[f"proc_noise_vel_ori_{k}"].get() for k in "xyz"],
            "meas_noise_pos": [self.kf_params[f"meas_noise_pos_{k}"].get() for k in "xyz"],
            "meas_noise_ori": [self.kf_params[f"meas_noise_ori_{k}"].get() for k in "xyz"]
        }
        processed_ts, processed_pos, processed_quat = self._run_preprocessing(buoy_df=original_df.copy())
        if processed_ts is None: return None
        
        _, smoothed_states = self._execute_smoother_logic(processed_ts, processed_pos, processed_quat, kf_params_dict)
        smooth_pos = smoothed_states[:, 0:3]
        smooth_vel = smoothed_states[:, 3:6]
        smooth_acc = smoothed_states[:, 6:9]
        smooth_quat = smoothed_states[:, 9:13]

        labels_pva, colors_plotly = ['X', 'Y', 'Z'], ['red', 'green', 'blue', 'purple']
        
        # Plot Position
        for i in range(3):
            fig.add_trace(go.Scatter(x=original_timestamps, y=original_positions[:, i], mode='markers', marker=dict(size=3, opacity=0.4), name=f'Raw Pos {labels_pva[i]}', line=dict(color=colors_plotly[i]), legendgroup='pos', legendgrouptitle_text='Position'), row=1, col=1)
            fig.add_trace(go.Scatter(x=processed_ts, y=smooth_pos[:, i], mode='lines', name=f'Smoothed Pos {labels_pva[i]}', line=dict(color=colors_plotly[i], width=2), legendgroup='pos'), row=1, col=1)

        # Plot Velocity, Acceleration, and Orientation
        for i in range(3): fig.add_trace(go.Scatter(x=processed_ts, y=smooth_vel[:, i], mode='lines', name=f'Smoothed Vel {labels_pva[i]}', line=dict(color=colors_plotly[i], width=1.5), legendgroup='vel', legendgrouptitle_text='Velocity'), row=2, col=1)
        for i in range(3): fig.add_trace(go.Scatter(x=processed_ts, y=smooth_acc[:, i], mode='lines', name=f'Smoothed Acc {labels_pva[i]}', line=dict(color=colors_plotly[i], width=1.5), legendgroup='acc', legendgrouptitle_text='Acceleration'), row=3, col=1)
        labels_quat = ['qx', 'qy', 'qz', 'qw']
        for i in range(4):
            fig.add_trace(go.Scatter(x=original_timestamps, y=original_quaternions[:, i], mode='markers', marker=dict(size=3, opacity=0.4), name=f'Raw {labels_quat[i]}', line=dict(color=colors_plotly[i]), legendgroup='ori', legendgrouptitle_text='Orientation'), row=4, col=1)
            fig.add_trace(go.Scatter(x=processed_ts, y=smooth_quat[:, i], mode='lines', name=f'Smoothed {labels_quat[i]}', line=dict(color=colors_plotly[i], width=2), legendgroup='ori'), row=4, col=1)

        fig.update_layout(height=1000, hovermode='x unified', legend_tracegroupgap=150, title_x=0.5)
        fig.update_yaxes(title_text="Position (m)", row=1, col=1); fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
        fig.update_yaxes(title_text="Accel (m/s^2)", row=3, col=1); fig.update_yaxes(title_text="Quat. Comp.", row=4, col=1, range=[-1.05, 1.05])
        fig.update_xaxes(title_text="Time (s)", row=4, col=1)
        
        return fig

    def _generate_noise_histogram_fig(self, buoy_id, static_df, start_t, end_t):
        """Generates the noise histogram figure for a given buoy's static data."""
        if static_df.empty or len(static_df) < 2:
            return None

        pos_devs_hist = None
        quats_hist = None

        static_pos = static_df[self.pos_cols_for_filter].values
        if static_pos.shape[0] >= 2:
            pos_devs_hist = static_pos - np.nanmean(static_pos, axis=0)

        static_quat = static_df[self.quat_cols_for_filter].values
        valid_static_quat = static_quat[~np.isnan(static_quat).any(axis=1)]
        if len(valid_static_quat) >= 2:
            quats_hist = np.array([q/np.linalg.norm(q) for q in valid_static_quat if np.linalg.norm(q) > 1e-9])

        return self._plot_noise_histograms(pos_devs_hist, quats_hist, start_t, end_t, buoy_id, return_fig=True)

    def _create_and_save_batch_report(self, report_path, html_snippets, buoy_ids):
        """Assembles and saves the final HTML batch report."""
        nav_links = ""
        for buoy_id in buoy_ids:
            nav_links += f'<li><a href="#buoy-{buoy_id}">Buoy {buoy_id}</a></li>'

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Processing Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: sans-serif; margin: 2em; }}
                h1, h2 {{ color: #333; }}
                .nav {{ list-style-type: none; padding: 0; margin-bottom: 2em; }}
                .nav li {{ display: inline-block; margin-right: 15px; }}
                .buoy-section {{ border-top: 2px solid #ccc; padding-top: 2em; margin-top: 2em; }}
            </style>
        </head>
        <body>
            <h1>Batch Processing Report</h1>
            <h2>Navigation</h2>
            <ul class="nav">{nav_links}</ul>
        """

        for i, snippet in enumerate(html_snippets):
            buoy_id = buoy_ids[i]
            html_content += f'<div id="buoy-{buoy_id}" class="buoy-section">{snippet}</div>'
            
        html_content += "</body></html>"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            webbrowser.open_new_tab(f'file://{os.path.realpath(report_path)}')
            return True
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to write or open the report file:\n{e}")
            return False

    def batch_process_all_buoys(self):
        if self.data_df is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return
            
        csv_save_path = os.path.join(save_dir, "processed_buoy_data_batch.csv")
        report_save_path = os.path.join(save_dir, "batch_report.html")

        self.batch_button.config(state="disabled")
        self.autotune_button.config(state="disabled")
        
        all_results_df = []
        html_snippets = []
        processed_buoy_ids = []

        buoy_ids = self.buoy_ids_in_file
        total_buoys = len(buoy_ids)
        self.progress_bar['maximum'] = total_buoys
        self.progress_bar['value'] = 0

        for i, buoy_id in enumerate(buoy_ids):
            self.status_var.set(f"Status: Processing buoy {i+1}/{total_buoys} (ID: {buoy_id})...")
            self.master.update_idletasks()

            try:
                processed_df, original_df, static_df, static_times = self._tune_and_smooth_single_buoy(buoy_id)
                
                if processed_df is not None:
                    all_results_df.append(processed_df)
                    processed_buoy_ids.append(buoy_id)

                    buoy_html = f"<h2>Report for Buoy ID: {buoy_id}</h2>"

                    hist_fig = self._generate_noise_histogram_fig(buoy_id, static_df, static_times[0], static_times[1])
                    if hist_fig:
                        buoy_html += hist_fig.to_html(full_html=False, include_plotlyjs=False)
                    
                    smoother_fig = self._generate_smoother_plot_fig(buoy_id, original_df, processed_df)
                    if smoother_fig:
                        buoy_html += smoother_fig.to_html(full_html=False, include_plotlyjs=False)
                    
                    html_snippets.append(buoy_html)
                else:
                    print(f"Warning: Skipping buoy {buoy_id} due to processing error or lack of data.")

            except Exception as e:
                print(f"FATAL ERROR processing buoy {buoy_id}: {e}")
                traceback.print_exc()
                messagebox.showwarning("Batch Error", f"An error occurred while processing buoy {buoy_id}:\n{e}\n\nSkipping to next buoy.")

            self.progress_bar['value'] = i + 1
            self.master.update_idletasks()

        if not all_results_df:
            messagebox.showinfo("Finished", "Batch processing complete, but no data was successfully processed.")
            self.status_var.set("Status: Batch processing finished with no results.")
        else:
            final_df = pd.concat(all_results_df, ignore_index=True)
            original_cols = self.data_df.columns.tolist()
            output_cols = [col for col in original_cols if col in final_df.columns]
            output_cols.extend([col for col in final_df.columns if col not in output_cols])
            final_df = final_df[output_cols]
            final_df.to_csv(csv_save_path, index=False, float_format='%.8f')
            
            self._create_and_save_batch_report(report_save_path, html_snippets, processed_buoy_ids)
            
            self.status_var.set(f"Status: Batch processing complete! Report saved to {report_save_path}")
            messagebox.showinfo("Finished", f"Successfully processed {len(all_results_df)} buoys.\nData saved to:\n{csv_save_path}\n\nReport saved to:\n{report_save_path}")

        self.batch_button.config(state="normal")
        self.autotune_button.config(state="normal")
        self.progress_bar['value'] = 0

    def _tune_and_smooth_single_buoy(self, buoy_id):
        """
        A helper function for the batch process that handles tuning and smoothing for one buoy.
        This function now measures R, tunes Q, smooths, and returns all necessary data for plotting.
        """
        original_buoy_df = self.data_df[self.data_df['Buoy_ID'] == buoy_id].copy()
        static_df_for_hist = pd.DataFrame()
        r_start_t, r_end_t = 0.0, 0.0

        self.status_var.set(f"Status: Measuring R for buoy {buoy_id}..."); self.master.update_idletasks()
        try:
            r_start_t = float(self.static_start_time_var.get())
            r_end_t = float(self.static_end_time_var.get())

            if r_start_t < r_end_t:
                static_mask = (original_buoy_df['Time_Rel_Sec'] >= r_start_t) & (original_buoy_df['Time_Rel_Sec'] <= r_end_t)
                static_df_for_hist = original_buoy_df[static_mask].copy()
                
                static_pos = static_df_for_hist[self.pos_cols_for_filter].values
                if len(static_pos) >= 2:
                    pos_std_calc = np.nanstd(static_pos, axis=0)
                    for i, k in enumerate("xyz"):
                        if not np.isnan(pos_std_calc[i]): self.kf_params[f"meas_noise_pos_{k}"].set(round(pos_std_calc[i], 4))
                
                static_quat = static_df_for_hist[self.quat_cols_for_filter].values
                valid_static_quat = static_quat[~np.isnan(static_quat).any(axis=1)]
                if len(valid_static_quat) >= 2:
                    norm_quats = [q/np.linalg.norm(q) for q in valid_static_quat if np.linalg.norm(q) > 1e-9]
                    if norm_quats:
                        mean_q = average_quaternions_weighted(norm_quats, [1.0]*len(norm_quats))
                        if not np.isnan(mean_q).any():
                            mean_q_obj = R.from_quat(mean_q)
                            err_vecs = [(R.from_quat(q_m) * mean_q_obj.inv()).as_rotvec() for q_m in norm_quats]
                            if err_vecs:
                                ori_std_calc = np.nanstd(np.array(err_vecs), axis=0)
                                for i, k in enumerate("xyz"):
                                    if not np.isnan(ori_std_calc[i]): self.kf_params[f"meas_noise_ori_{k}"].set(round(ori_std_calc[i], 4))
                
                measured_r_pos = [self.kf_params[f'meas_noise_pos_{k}'].get() for k in 'xyz']
                measured_r_ori = [self.kf_params[f'meas_noise_ori_{k}'].get() for k in 'xyz']
                print(f"Buoy {buoy_id} R updated to Pos: {[f'{v:.4f}' for v in measured_r_pos]}, Ori: {[f'{v:.4f}' for v in measured_r_ori]}")
            else:
                 print(f"Warning for Buoy {buoy_id}: Invalid static time for R measurement. Using current/previous R values.")
        except (ValueError, IndexError):
            print(f"Warning for Buoy {buoy_id}: Could not parse static time for R measurement. Using current/previous R values.")

        processed_data = self._run_preprocessing(buoy_df=original_buoy_df.copy())
        if processed_data[0] is None:
            return None, None, None, None

        timestamps, positions, quaternions = processed_data

        try:
            start_t = float(self.tune_start_time_var.get())
            end_t_str = self.tune_end_time_var.get()
            end_t = float(end_t_str) if end_t_str.strip() else np.inf
        except ValueError: start_t, end_t = 0.0, np.inf

        mask = (timestamps >= start_t) & (timestamps <= end_t)
        if np.sum(mask) < 20:
            print(f"Warning for Buoy {buoy_id}: Not enough data for Q-tuning ({np.sum(mask)} points). Using default Q values.")
            optimal_q_params = np.array(list(DEFAULT_KF_PROCESS_NOISE_ACC) + list(DEFAULT_KF_PROCESS_NOISE_VEL_ORI))
        else:
            self.status_var.set(f"Status: Tuning Q for buoy {buoy_id}..."); self.master.update_idletasks()
            initial_guess = np.array([self.kf_params[f"proc_noise_acc_{k}"].get() for k in "xyz"] +
                                     [self.kf_params[f"proc_noise_vel_ori_{k}"].get() for k in "xyz"])
            self.tune_iteration = 0
            result = minimize(
                fun=self._cost_function_nll,
                x0=initial_guess,
                args=(timestamps[mask], positions[mask], quaternions[mask]),
                method='L-BFGS-B', bounds=[(1e-2, 10.0)]*6, options={'ftol': 1e-10}
            )
            if result.success:
                optimal_q_params = result.x
                print(f"Buoy {buoy_id} Q tuned. Optimal Q: {np.round(optimal_q_params, 4)}")
            else:
                print(f"Warning for Buoy {buoy_id}: Q-Tuning failed. Using default Q values.")
                optimal_q_params = np.array(list(DEFAULT_KF_PROCESS_NOISE_ACC) + list(DEFAULT_KF_PROCESS_NOISE_VEL_ORI))
        
        self.status_var.set(f"Status: Smoothing data for buoy {buoy_id}..."); self.master.update_idletasks()
        kf_params_for_run = {
            "proc_noise_acc": optimal_q_params[0:3],
            "proc_noise_vel_ori": optimal_q_params[3:6],
            "meas_noise_pos": [self.kf_params[f"meas_noise_pos_{k}"].get() for k in "xyz"],
            "meas_noise_ori": [self.kf_params[f"meas_noise_ori_{k}"].get() for k in "xyz"]
        }
        smoothed_timestamps, smoothed_states = self._execute_smoother_logic(timestamps, positions, quaternions, kf_params_for_run)
        
        # ### MODIFICATION START ###
        # Define new column names for velocity and acceleration
        vel_cols = ['CoG_Vel_X_Rel', 'CoG_Vel_Y_Rel', 'CoG_Vel_Z_Rel']
        acc_cols = ['CoG_Acc_X_Rel', 'CoG_Acc_Y_Rel', 'CoG_Acc_Z_Rel']

        smoothed_df = pd.DataFrame({
            'Time_Rel_Sec': smoothed_timestamps,
            # Smoothed Position
            self.pos_cols_for_filter[0]: smoothed_states[:, 0],
            self.pos_cols_for_filter[1]: smoothed_states[:, 1],
            self.pos_cols_for_filter[2]: smoothed_states[:, 2],
            # Smoothed Velocity
            vel_cols[0]: smoothed_states[:, 3],
            vel_cols[1]: smoothed_states[:, 4],
            vel_cols[2]: smoothed_states[:, 5],
            # Smoothed Acceleration
            acc_cols[0]: smoothed_states[:, 6],
            acc_cols[1]: smoothed_states[:, 7],
            acc_cols[2]: smoothed_states[:, 8],
            # Smoothed Orientation
            self.quat_cols_for_filter[0]: smoothed_states[:, 9],
            self.quat_cols_for_filter[1]: smoothed_states[:, 10],
            self.quat_cols_for_filter[2]: smoothed_states[:, 11],
            self.quat_cols_for_filter[3]: smoothed_states[:, 12],
        })
        # ### MODIFICATION END ###

        cols_to_drop_from_orig = self.pos_cols_for_filter + self.quat_cols_for_filter + ['Timestamp_sec']
        final_buoy_df = pd.merge_asof(
            smoothed_df.sort_values('Time_Rel_Sec'),
            original_buoy_df.drop(columns=cols_to_drop_from_orig, errors='ignore').sort_values('Time_Rel_Sec'),
            on='Time_Rel_Sec',
            direction='nearest'
        )

        return final_buoy_df, original_buoy_df, static_df_for_hist, (r_start_t, r_end_t)

if __name__ == "__main__":
    root = tk.Tk()
    app = KFTunerApp(root)
    root.mainloop()