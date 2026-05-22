# ==============================================================================
#      *** 1. NECESSARY IMPORTS ***
# ==============================================================================
import pandas as pd
import numpy as np
import re
import os
import logging
from pathlib import Path
from datetime import datetime
import threading
import traceback

# --- Scientific Computing & Plotting ---
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft, fftfreq

# --- GUI Components ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    # tkinterdnd2 is an optional dependency for drag-and-drop functionality
    import tkinterdnd2
except ImportError:
    tkinterdnd2 = None


# ==============================================================================
#      *** 2. PHYSICAL CONSTANTS & CONFIGURATION ***
# ==============================================================================
# --- Hydrodynamic Parameters ---
RHO_WATER = 1000                # Density of water (kg/m^3)
G = 9.81                        # Acceleration due to gravity (m/s^2)
DEFAULT_EPSILON = 1e-9          # Epsilon for monotonic unwrap
WEC_ARRAY_WIDTH = 0.25          # Overall width of the buoy array (meters)
BUOY_DIAMETER = WEC_ARRAY_WIDTH # Assuming array width is buoy diameter for accordion calc
SINGLE_BUOY_MASS = 0.5          # kg


# --- Data Sync Trigger Parameters ---
PADDLE_TRIGGER_ON_VOLTAGE = -2.5
LED_TRIGGER_ON_THRESHOLD = 0.5


# ==============================================================================
#      *** 3. NEW DATA IMPORT & SYNCHRONIZATION BACKEND ***
# ==============================================================================
class DataSynchronizer:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.dof_df = None
        self.gauge_df = None
        self.pto_df = None
        self.metadata = {}
        self.original_dof_columns = []
        self.original_gauge_columns = []
        self.original_pto_columns = []

    def _log(self, message):
        # In a real app, this would use the logging module. For this script, print is sufficient.
        if self.verbose:
            print(f"SYNC_LOG: {message}")

    def _find_gauge_data_start(self, filepath):
        try:
            with open(filepath, 'r', errors='ignore') as f:
                for i, line in enumerate(f):
                    if re.match(r'^Time\s+', line) and '\t' in line:
                        self._log(f"Detected gauge data header on line {i+1}. Data starts on line {i+3}.")
                        return i + 2
                    if i > 50: break
        except Exception as e:
            self._log(f"Error while searching for gauge header: {e}")
        self._log("Could not reliably detect gauge header. Assuming data starts on line 6.")
        return 6

    def _clean_and_uniquify_time(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        df = df.rename(columns={time_col: 'Time_s'})
        df['Time_s'] = pd.to_numeric(df['Time_s'], errors='coerce')
        df.dropna(subset=['Time_s'], inplace=True)
        if df.empty: return pd.DataFrame()

        # Standardize Buoy ID column to 'Buoy_ID' for the analysis classes
        buoy_id_col = next((c for c in df.columns if 'buoy' in c.lower() and 'id' in c.lower()), None)
        if buoy_id_col and buoy_id_col != 'Buoy_ID':
            df.rename(columns={buoy_id_col: 'Buoy_ID'}, inplace=True)

        grouping_cols = ['Time_s']
        if 'Buoy_ID' in df.columns:
            grouping_cols.append('Buoy_ID')

        if df.duplicated(subset=grouping_cols).any():
            self._log(f"Handling duplicates by aggregating. Grouping by: {', '.join(grouping_cols)}")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            object_cols = df.select_dtypes(include='object').columns.tolist()
            agg_dict = {col: 'mean' for col in numeric_cols if col not in grouping_cols}
            agg_dict.update({col: 'first' for col in object_cols if col not in grouping_cols})
            df = df.groupby(grouping_cols, as_index=False).agg(agg_dict)

        return df.sort_values(by=grouping_cols).reset_index(drop=True)

    def load_dof_data(self, filepath: Path):
        self._log(f"--- Loading DOF Data from: {filepath.name} ---")
        try:
            df = pd.read_csv(filepath)
            self.original_dof_columns = [c for c in df.columns if c != 'Time_Rel_Sec']
            df = self._clean_and_uniquify_time(df, time_col='Time_Rel_Sec')
            self.dof_df = df
            self._log("Successfully loaded DOF data.")
        except Exception as e:
            print(f"ERROR: Failed to process DOF file {filepath.name}. Error: {e}")

    def load_gauge_data(self, filepath: Path):
        self._log(f"--- Loading Gauge Data from: {filepath.name} ---")
        try:
            start_line = self._find_gauge_data_start(filepath)
            with open(filepath, 'r', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        if match := re.search(r'Freq=([0-9.]+)', line): self.metadata['Frequency_Hz'] = float(match.group(1))
                        if match := re.search(r'Amp=([0-9.]+)', line): self.metadata['Amplitude_m'] = float(match.group(1))
                    else: break
            df = pd.read_csv(filepath, sep='\t', skiprows=start_line, header=None, low_memory=False)
            num_gauge_cols = 8
            gauge_cols = [f'Gauge_{i}' for i in range(1, num_gauge_cols + 1)]
            cols_to_use_indices = [0] + list(range(df.shape[1] - num_gauge_cols, df.shape[1]))
            df = df.iloc[:, cols_to_use_indices]
            df.columns = [0] + gauge_cols
            self.original_gauge_columns = gauge_cols
            df = self._clean_and_uniquify_time(df, time_col=0)
            self.gauge_df = df
            self._log("Successfully loaded Gauge data.")
        except Exception as e:
            print(f"ERROR: Failed to process gauge file {filepath.name}. Error: {e}")

    def load_pto_data(self, filepath: Path):
        self._log(f"--- Loading PTO Data from: {filepath.name} ---")
        try:
            df_raw = pd.read_csv(filepath, sep='\t', header=0, engine='python')
            df = pd.DataFrame({'Time_Raw': df_raw.index,'TriggerPaddle_V': df_raw.iloc[:, 0],'Flow_lpm': df_raw.iloc[:, 3],'Pressure_bar': df_raw.iloc[:, 4]})
            df = self._clean_and_uniquify_time(df, time_col='Time_Raw')
            df['Pressure_Pa'] = df['Pressure_bar'] * 100000
            df['Flow_Rate_m3s'] = df['Flow_lpm'] / 60000
            df['Sync_Signal_PF'] = df['TriggerPaddle_V'] < PADDLE_TRIGGER_ON_VOLTAGE
            self.original_pto_columns = [c for c in df.columns if c != 'Time_s']
            self.pto_df = df
            self._log("Successfully loaded PTO data.")
        except Exception as e:
            print(f"ERROR: Failed to process PTO file {filepath.name}. Error: {e}")

    def synchronize(self):
        self._log("\n" + "="*50 + "\n--- STARTING FULL DATA SYNCHRONIZATION (GAUGE as Master) ---\n" + "="*50)

        if self.gauge_df is None or self.gauge_df.empty:
            print("ERROR: GAUGE file is required for synchronization but is not loaded.")
            return pd.DataFrame(), self.metadata

        master_start_time = self.gauge_df['Time_s'].min()
        self._log(f"GAUGE file defines master timeline. Time Zero = {master_start_time:.4f}s.")

        dof_aligned = self.dof_df.copy() if self.dof_df is not None else None
        pto_aligned = self.pto_df.copy() if self.pto_df is not None else None

        if dof_aligned is not None:
            sync_window = dof_aligned[dof_aligned['LED_On'] > LED_TRIGGER_ON_THRESHOLD]
            if not sync_window.empty:
                dof_sync_time = sync_window['Time_s'].min()
                offset = dof_sync_time - master_start_time
                dof_aligned['Time_s'] -= offset
                self._log(f"DOF sync event found at {dof_sync_time:.4f}s. Shifting DOF timeline by {-offset:.4f}s.")
            else:
                print("WARNING: No sync signal ('LED_On') found in DOF file. It will be excluded.")
                dof_aligned = None

        if pto_aligned is not None:
            sync_window = pto_aligned[pto_aligned['Sync_Signal_PF'] == True]
            if not sync_window.empty:
                pto_sync_time = sync_window['Time_s'].min()
                offset = pto_sync_time - master_start_time
                pto_aligned['Time_s'] -= offset
                self._log(f"PTO sync event found at {pto_sync_time:.4f}s. Shifting PTO timeline by {-offset:.4f}s.")
            else:
                print("WARNING: No sync signal ('Trigger') found in PTO file. It will be excluded.")
                pto_aligned = None

        self._log("Merging all aligned data streams...")
        final_df = self.gauge_df.copy()
        if dof_aligned is not None:
            final_df = pd.merge(final_df, dof_aligned, on='Time_s', how='outer', sort=True)
        if pto_aligned is not None:
            final_df = pd.merge(final_df, pto_aligned, on='Time_s', how='outer', sort=True)

        self._log("Performing final interpolation...")
        if 'Buoy_ID' in final_df.columns:
            final_df['Buoy_ID'] = final_df['Buoy_ID'].ffill().bfill()

        dof_cols = [c for c in self.original_dof_columns if c in final_df.columns and c != 'Buoy_ID']
        gauge_cols = [c for c in self.original_gauge_columns if c in final_df.columns]
        pto_cols = [c for c in self.original_pto_columns if c in final_df.columns]

        if 'Buoy_ID' in final_df.columns and dof_cols:
            self._log(f"Interpolating {len(dof_cols)} DOF columns per buoy...")
            final_df[dof_cols] = final_df.groupby('Buoy_ID')[dof_cols].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )

        global_cols = gauge_cols + pto_cols
        if global_cols:
            self._log(f"Interpolating {len(global_cols)} global (Gauge/PTO) columns...")
            final_df[global_cols] = final_df[global_cols].interpolate(method='linear', limit_direction='both')

        final_df = final_df[
            (final_df['Time_s'] >= self.gauge_df['Time_s'].min()) &
            (final_df['Time_s'] <= self.gauge_df['Time_s'].max())
        ].copy()

        final_df.dropna(subset=['Buoy_ID'] if 'Buoy_ID' in final_df.columns else ['Time_s'], inplace=True)
        if 'Buoy_ID' in final_df.columns:
            final_df['Buoy_ID'] = pd.to_numeric(final_df['Buoy_ID'], errors='coerce').round().astype('Int64')

        self._log("Selecting and reordering final columns...")
        desired_columns = [
            'Time_s', 'Buoy_ID',
            'Gauge_1', 'Gauge_2', 'Gauge_3', 'Gauge_4', 'Gauge_5', 'Gauge_6', 'Gauge_7', 'Gauge_8',
            'CoG_Pos_X_Rel', 'CoG_Pos_Y_Rel', 'CoG_Pos_Z_Rel',
            'CoG_Vel_X_Rel', 'CoG_Vel_Y_Rel', 'CoG_Vel_Z_Rel',
            'CoG_Acc_X_Rel', 'CoG_Acc_Y_Rel', 'CoG_Acc_Z_Rel',
            'CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel',
            'Pressure_bar', 'Pressure_Pa', 'Flow_lpm', 'Flow_Rate_m3s'
        ]
        existing_desired_columns = [col for col in desired_columns if col in final_df.columns]
        final_df = final_df[existing_desired_columns]

        # Handle case where optional files were not loaded
        if 'Pressure_Pa' not in final_df.columns:
            final_df['Pressure_Pa'] = 0
            final_df['Flow_Rate_m3s'] = 0
            self._log("PTO data not present; Power/Flow columns set to zero.")

        self._log(f"--- FINAL FORMATTING COMPLETE! Final Shape: {final_df.shape} ---")
        return final_df, self.metadata


# ==============================================================================
#      *** 4. ANALYSIS & REPORTING CLASSES ***
# ==============================================================================
class ExperimentalSetup:
    """
    A data structure to store the physical layout of the experimental setup
    and calculate distances between points.
    """
    def __init__(self, positions_dict):
        self.positions = {str(name).lower(): np.array(coords) for name, coords in positions_dict.items()}
        print("ExperimentalSetup initialized with the following points:", list(self.positions.keys()))

    def get_position(self, point_name):
        lookup_key = str(point_name).lower()
        if lookup_key not in self.positions:
            raise KeyError(f"Error: Point '{point_name}' (as '{lookup_key}') not found in the setup.")
        return self.positions[lookup_key]

class HydrodynamicAnalysis:
    """
    Performs extended hydrodynamic analysis on synchronized WEC data.
    """
    def __init__(self, synchronized_df, metadata, experimental_setup, transient_duration=5.0):
        if synchronized_df.empty:
            raise ValueError("Input DataFrame is empty. Cannot perform analysis.")
        self.start_time = synchronized_df['Time_s'].min() + transient_duration
        self.df = synchronized_df[synchronized_df['Time_s'] >= self.start_time].copy()
        if self.df.empty:
             raise ValueError(f"No data remains after removing transient duration of {transient_duration}s. Analysis cannot continue.")
        print(f"\nAnalysis Warning: Excluding first {transient_duration}s as transient regime.")
        print(f"Analysis will run on data from t={self.start_time:.2f}s onwards.")
        self.metadata = metadata
        self.setup = experimental_setup
        self.buoy_ids = sorted([int(id) for id in self.df['Buoy_ID'].unique()])
        self.time = self.df['Time_s'].unique()
        self.fs = 1 / (self.time[1] - self.time[0]) if len(self.time) > 1 else 1.0
        print(f"Analysis class initialized for Buoy IDs: {self.buoy_ids}")
        print(f"Detected sampling frequency: {self.fs:.2f} Hz")
        self.results = {'transient_duration': transient_duration}

    @staticmethod
    def _highpass_filter(data, cutoff_hz, fs, order=2):
        """Applies a zero-phase high-pass Butterworth filter."""
        nyquist = 0.5 * fs
        if cutoff_hz >= nyquist:
            print(f"Warning: Cutoff frequency ({cutoff_hz} Hz) is near or above Nyquist frequency ({nyquist} Hz). "
                  "Filter will have little to no effect.")
            return data
        normal_cutoff = cutoff_hz / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    @staticmethod
    def _monotonic_unwrap(phases, increasing=False):
        """Unwraps phases to be strictly monotonic for robust linear fitting."""
        unwrapped = np.unwrap(phases)
        for i in range(1, len(unwrapped)):
            if increasing:
                while unwrapped[i] <= unwrapped[i-1] + DEFAULT_EPSILON: unwrapped[i] += 2 * np.pi
            else:
                while unwrapped[i] >= unwrapped[i-1] - DEFAULT_EPSILON: unwrapped[i] -= 2 * np.pi
        return unwrapped

    def analyze_monochromatic_wave_properties(self, selected_gauges=None):
        print("\n--- Analyzing Wave Properties & Relative Motion Phases ---")
        if selected_gauges is None:
            selected_gauges = sorted([col for col in self.df.columns if col.startswith('Gauge_')])

        gauge_locations = []
        wave_phases = []
        wave_amplitudes = []

        df_single_buoy_view = self.df[self.df['Buoy_ID'] == self.buoy_ids[0]]
        if df_single_buoy_view.empty or not selected_gauges:
            print("  - Error: No data available for wave property analysis. Skipping.")
            self.results['wave_properties'] = {}
            return {}

        N = len(df_single_buoy_view)
        # Determine dominant frequency from the primary incident gauge for consistency
        yf_wave = fft(df_single_buoy_view[INCIDENT_WAVE_GAUGE].to_numpy())
        xf = fftfreq(N, 1.0 / self.fs)[:N // 2]
        max_idx = np.argmax(np.abs(yf_wave[1:N // 2])) + 1
        dominant_freq = xf[max_idx]

        for gauge_col in selected_gauges:
            try:
                location = self.setup.get_position(gauge_col)[0] # We only need X-position
                yf_gauge = fft(df_single_buoy_view[gauge_col].to_numpy())
                wave_phases.append(np.angle(yf_gauge[max_idx]))
                wave_amplitudes.append(2.0 / N * np.abs(yf_gauge[max_idx]))
                gauge_locations.append(location)
            except KeyError:
                print(f"  - Warning: Gauge '{gauge_col}' found in data but not defined in ExperimentalSetup. Skipping analysis for this gauge.")
                continue

        if len(gauge_locations) < 2:
            raise ValueError(f"Error: Need at least 2 gauges with known positions to determine wavenumber, but only found {len(gauge_locations)}. Please check your ExperimentalSetup definition.")

        unwrapped_phases = self._monotonic_unwrap(np.array(wave_phases), increasing=False)
        loc_phase_pairs = sorted(zip(gauge_locations, unwrapped_phases))
        sorted_locations = np.array([p[0] for p in loc_phase_pairs])
        sorted_phases = np.array([p[1] for p in loc_phase_pairs])
        k = abs(np.polyfit(sorted_locations, sorted_phases, 1)[0])

        self.results['wave_properties'] = {
            'frequency': dominant_freq, 'amplitude': np.mean(wave_amplitudes),
            'wavenumber': k, 'wavelength': 2 * np.pi / k if k > 0 else np.inf,
        }

        print("  - Calculating relative heave motion phases...")
        motion_phases = {}
        ref_phase = 0
        for i, buoy_id in enumerate(self.buoy_ids):
            heave_signal = self.df[self.df['Buoy_ID'] == buoy_id]['CoG_Pos_Y_Rel'].to_numpy()
            yf_motion = fft(heave_signal)
            current_phase = np.angle(yf_motion[max_idx])

            if buoy_id == REFERENCE_BUOY_ID:
                ref_phase = current_phase
            motion_phases[buoy_id] = current_phase

        relative_phases_deg = {buoy_id: np.rad2deg(np.unwrap([ref_phase, p])[1] - ref_phase)
                               for buoy_id, p in motion_phases.items()}

        self.results['relative_heave_phase_deg'] = relative_phases_deg
        print("Wave and phase analysis complete.")
        return self.results['wave_properties']