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
REFERENCE_BUOY_ID = 1           # Buoy ID used as the zero-phase reference

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
        self.results = {}

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

    def calculate_wave_to_buoy_phase(self):
        """Calculates the phase difference between the incident wave and each buoy's heave motion."""
        print("\n--- Calculating Wave-to-Buoy Phase Difference ---")
        if 'wave_properties' not in self.results:
            self.analyze_monochromatic_wave_properties()
        props = self.results['wave_properties']
        k = props['wavenumber']
        df_single_buoy_view = self.df[self.df['Buoy_ID'] == self.buoy_ids[0]]
        N = len(df_single_buoy_view)
        wave_fft_full = fft(df_single_buoy_view[INCIDENT_WAVE_GAUGE].to_numpy())
        max_idx = np.argmax(np.abs(wave_fft_full[1:N//2])) + 1
        phase_at_gauge = np.angle(wave_fft_full[max_idx])
        gauge_pos_x = self.setup.get_position(INCIDENT_WAVE_GAUGE)[0]
        phase_lags = {}
        for buoy_id in self.buoy_ids:
            buoy_df = self.df[self.df['Buoy_ID'] == buoy_id]
            heave_signal = buoy_df['CoG_Pos_Y_Rel'].to_numpy()
            phase_buoy_motion = np.angle(fft(heave_signal)[max_idx])
            buoy_pos_x = buoy_df['CoG_Pos_X_Rel'].mean()
            distance_from_gauge = buoy_pos_x - gauge_pos_x
            phase_wave_at_buoy = phase_at_gauge - k * distance_from_gauge
            phase_lag = np.unwrap([phase_wave_at_buoy, phase_buoy_motion])[1] - phase_wave_at_buoy
            phase_lags[buoy_id] = np.rad2deg(phase_lag)
        self.results['wave_to_buoy_phase_deg'] = phase_lags
        print("Wave-to-buoy phase calculation complete.")
        return phase_lags

    def calculate_power_from_mono_analysis(self, water_depth=np.inf):
        print("\n--- Calculating Incident Wave Power (J) from Monochromatic Results ---")
        if 'wave_properties' not in self.results:
            self.analyze_monochromatic_wave_properties()
        props = self.results['wave_properties']
        H = props['amplitude'] * 2
        T = 1 / props['frequency'] if props['frequency'] > 0 else np.inf
        J = (RHO_WATER * G**2 / (32 * np.pi)) * H**2 * T
        self.results['wave_power_per_meter'] = J
        print(f"Derived Wave Height (H): {H:.4f} m, Period (T): {T:.3f} s")
        print(f"Average Incident Wave Power (J): {J:.2f} W/m")
        return J

    def calculate_raos(self):
        print("\n--- Calculating Response Amplitude Operators (RAOs) ---")
        if 'wave_properties' not in self.results:
            self.analyze_monochromatic_wave_properties()
        raos = {}
        wave_amplitude = self.results['wave_properties']['amplitude']
        dominant_freq = self.results['wave_properties']['frequency']
        cutoff_frequency = dominant_freq / 5.0
        print(f"Applying high-pass filter with cutoff at {cutoff_frequency:.3f} Hz to remove drift.")
        motion_types = {'Heave': 'CoG_Pos_Y_Rel', 'Surge': 'CoG_Pos_X_Rel', 'Sway': 'CoG_Pos_Z_Rel'}
        for buoy_id in self.buoy_ids:
            buoy_df = self.df[self.df['Buoy_ID'] == buoy_id]
            raos[buoy_id] = {}
            for motion_name, col_name in motion_types.items():
                if col_name not in buoy_df.columns: continue
                motion_signal = buoy_df[col_name].to_numpy()
                filtered_motion_signal = self._highpass_filter(motion_signal, cutoff_frequency, self.fs)
                motion_amplitude = np.std(filtered_motion_signal) * np.sqrt(2)
                rao_value = motion_amplitude / wave_amplitude if wave_amplitude > 1e-6 else 0
                raos[buoy_id][motion_name] = rao_value
                print(f"  - Calculated {motion_name} RAO for Buoy ID {buoy_id}: {rao_value:.2f}")
        self.results['raos'] = raos
        print("RAO calculation complete.")
        return raos

    def calculate_mean_drift(self):
        print("\n--- Calculating Mean Drift Displacements ---")
        drift_results = {}
        for buoy_id in self.buoy_ids:
            buoy_df = self.df[self.df['Buoy_ID'] == buoy_id]
            drift_results[buoy_id] = {
                'Surge_Drift_m': buoy_df['CoG_Pos_X_Rel'].mean(),
                'Sway_Drift_m':  buoy_df['CoG_Pos_Z_Rel'].mean(),
                'Heave_Drift_m': buoy_df['CoG_Pos_Y_Rel'].mean()
            }
        self.results['mean_drift'] = drift_results
        print("Mean drift calculation complete.")
        return drift_results

    def calculate_system_kinetic_energy(self):
        print("\n--- Calculating System Kinetic Energy ---")
        vel_cols = ['CoG_Vel_X_Rel', 'CoG_Vel_Y_Rel', 'CoG_Vel_Z_Rel']
        if not all(col in self.df.columns for col in vel_cols):
            print("Warning: Velocity columns not found. Skipping kinetic energy calculation.")
            return None
        self.df['speed_sq'] = self.df[vel_cols[0]]**2 + self.df[vel_cols[1]]**2 + self.df[vel_cols[2]]**2
        total_ke_series = self.df.groupby('Time_s')['speed_sq'].sum() * 0.5 * SINGLE_BUOY_MASS
        mean_ke = total_ke_series.mean()
        self.results['mean_system_ke'] = mean_ke
        print(f"Average System Kinetic Energy: {mean_ke:.4f} Joules")
        return mean_ke

    def analyze_surge_accordion_effect(self):
        """Calculates the GAP oscillation between adjacent buoys."""
        print("\n--- Analyzing Surge Accordion Effect (Gap Oscillations) ---")
        if len(self.buoy_ids) < 2:
            print("  Warning: Need at least two buoys to analyze relative motion. Skipping.")
            return None
        accordion_results = {}
        sorted_buoy_ids = sorted(self.buoy_ids)
        for i in range(len(sorted_buoy_ids) - 1):
            buoy_id_1 = sorted_buoy_ids[i]
            buoy_id_2 = sorted_buoy_ids[i+1]
            pair_name = f"Pair_{buoy_id_1}-{buoy_id_2}"
            surge_1 = self.df[self.df['Buoy_ID'] == buoy_id_1]['CoG_Pos_X_Rel'].to_numpy()
            surge_2 = self.df[self.df['Buoy_ID'] == buoy_id_2]['CoG_Pos_X_Rel'].to_numpy()
            min_len = min(len(surge_1), len(surge_2))
            surge_1, surge_2 = surge_1[:min_len], surge_2[:min_len]
            center_to_center_distance = surge_2 - surge_1
            relative_gap = center_to_center_distance - BUOY_DIAMETER
            mean_gap = np.mean(relative_gap)
            min_gap = np.min(relative_gap)
            max_gap = np.max(relative_gap)
            oscillation_signal = center_to_center_distance - np.mean(center_to_center_distance)
            yf = fft(oscillation_signal)
            N = len(yf)
            xf = fftfreq(N, 1.0 / self.fs)[:N // 2]
            max_idx = np.argmax(np.abs(yf[1:N // 2])) + 1
            accordion_freq = xf[max_idx]
            accordion_amplitude = (2.0 / N) * np.abs(yf[max_idx])
            accordion_results[pair_name] = {
                'mean_gap_m': mean_gap, 'min_gap_m': min_gap, 'max_gap_m': max_gap,
                'oscillation_amplitude_m': accordion_amplitude,
                'oscillation_frequency_hz': accordion_freq,
                'time_series_gap_data': relative_gap
            }
            print(f"  Analysis for {pair_name}: Min Gap = {min_gap:.3f}m, Oscillation Amp = {accordion_amplitude:.4f}m")
        self.results['accordion_effect'] = accordion_results
        print("Accordion effect analysis complete.")
        return accordion_results

    def plot_drift_removal_diagnostic(self, buoy_id, motion_type='Heave'):
        """Plots a comparison of original, filtered, and drift signals."""
        if 'raos' not in self.results: print("Error: Run calculate_raos() first."); return
        motion_map = {'Heave': 'CoG_Pos_Y_Rel', 'Surge': 'CoG_Pos_X_Rel', 'Sway': 'CoG_Pos_Z_Rel'}
        col_name = motion_map.get(motion_type)
        if not col_name: print(f"Error: Motion type '{motion_type}' not recognized."); return
        buoy_df = self.df[self.df['Buoy_ID'] == buoy_id]
        if buoy_df.empty: print(f"Error: No data for Buoy ID {buoy_id}."); return
        original_signal = buoy_df[col_name].to_numpy()
        time_vector = buoy_df['Time_s'].to_numpy()
        dominant_freq = self.results['wave_properties']['frequency']
        cutoff_frequency = dominant_freq / 5.0
        filtered_signal = self._highpass_filter(original_signal, cutoff_frequency, self.fs)
        drift_signal = original_signal - filtered_signal
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(time_vector, original_signal, label='1. Original Signal', color='blue', alpha=0.7)
        ax.plot(time_vector, drift_signal, label='2. Removed Drift (Low-pass)', color='red', linestyle='--', lw=2.5)
        ax.plot(time_vector, filtered_signal, label='3. Filtered Signal (High-pass)', color='green', alpha=0.9)
        ax.set_title(f'Drift Removal Diagnostic for Buoy {buoy_id} - {motion_type} Motion', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12); ax.set_ylabel('Position (m)', fontsize=12)
        ax.legend(fontsize=11); ax.grid(True, which='both', linestyle='--', linewidth=0.5); plt.show()

    def plot_accordion_effect(self):
        """Generates plots to visualize the surge GAP oscillation."""
        if 'accordion_effect' not in self.results or not self.results['accordion_effect']:
            print("No accordion effect data to plot. Run analyze_surge_accordion_effect() first."); return
        print("\n--- Generating Accordion Effect (Gap) Diagnostic Plots ---")
        time_vector = self.df['Time_s'].unique()
        for pair_name, data in self.results['accordion_effect'].items():
            gap_signal = data['time_series_gap_data']; min_len = min(len(time_vector), len(gap_signal))
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=False, gridspec_kw={'height_ratios': [2, 1]})
            fig.suptitle(f'Gap Analysis for Buoy Pair: {pair_name}', fontsize=18)
            ax1.plot(time_vector[:min_len], gap_signal[:min_len], label='Instantaneous Gap', color='purple')
            ax1.axhline(data['mean_gap_m'], color='red', linestyle='--', label=f"Mean Gap: {data['mean_gap_m']:.3f} m")
            if data['min_gap_m'] < 0: ax1.axhspan(data['min_gap_m'], 0, color='red', alpha=0.2, label='Collision Occurred')
            ax1.set_title('Time-Domain View: Gap Between Buoy Surfaces'); ax1.set_ylabel('Gap (m)'); ax1.legend(); ax1.grid(True, which='both')
            N = len(gap_signal); yf = fft(gap_signal - data['mean_gap_m']); xf = fftfreq(N, 1.0 / self.fs)
            ax2.plot(xf[:N//2], (2.0/N) * np.abs(yf[:N//2]), color='darkorange')
            ax2.axvline(data['oscillation_frequency_hz'], color='black', ls='--', label=f"Dom Freq: {data['oscillation_frequency_hz']:.3f} Hz")
            ax2.set_title('Frequency-Domain View: Gap Oscillation Spectrum'); ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Gap Oscillation Amp (m)')
            ax2.set_xlim(0, xf[N//2 -1] if N > 2 else 1); ax2.legend(); ax2.grid(True)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    def summarize_results(self):
        print("\n" + "="*70 + "\n                 *** EXTENDED ANALYSIS RESULTS SUMMARY ***\n" + "="*70)
        if 'wave_properties' in self.results:
            props = self.results['wave_properties']
            print(f"Incident Wave Properties:\n  - Dominant Frequency: {props['frequency']:.3f} Hz\n  - Mean Amplitude: {props['amplitude']:.4f} m\n  - Wavenumber (k): {props['wavenumber']:.3f} rad/m")
        if 'wave_power_per_meter' in self.results: print(f"  - Derived Wave Power (J): {self.results['wave_power_per_meter']:.2f} W/m")
        print("-" * 70)
        if 'raos' in self.results:
            print("Response Amplitude Operators (RAO):")
            for buoy_id, rao_dict in self.results['raos'].items(): print(f"  Buoy {buoy_id} -> " + ", ".join([f"{m}: {v:.2f}" for m, v in rao_dict.items()]))
        print("-" * 70)
        if 'wave_to_buoy_phase_deg' in self.results:
            print(f"Wave-to-Buoy Heave Phase Lag (°):\n  " + " | ".join([f"Buoy {bid}: {p:.1f}°" for bid, p in self.results['wave_to_buoy_phase_deg'].items()]))
        print("-" * 70)
        if 'relative_heave_phase_deg' in self.results:
            print(f"Buoy-to-Buoy Relative Heave Phase (°, ref: Buoy {REFERENCE_BUOY_ID}):\n  " + " | ".join([f"Buoy {bid}: {p:.1f}°" for bid, p in self.results['relative_heave_phase_deg'].items()]))
        print("-" * 70)
        if 'mean_drift' in self.results:
            print("Mean Drift Displacement (m):")
            for buoy_id, drift_dict in self.results['mean_drift'].items(): print(f"  Buoy {buoy_id} -> Surge: {drift_dict['Surge_Drift_m']:.4f}, Sway: {drift_dict['Sway_Drift_m']:.4f}")
        print("-" * 70)
        if 'accordion_effect' in self.results:
            print("Accordion Effect (Relative Surge Gap Oscillation):")
            for pair, data in self.results['accordion_effect'].items(): print(f"  {pair}: Mean Gap: {data['mean_gap_m']:.3f} m | Min Gap: {data['min_gap_m']:.3f} m (Collision if < 0) | Osc. Amp: {data['oscillation_amplitude_m']:.4f} m")
        print("-" * 70)
        if 'mean_system_ke' in self.results: print(f"System Energy Proxy:\n  - Average Total Kinetic Energy: {self.results['mean_system_ke']:.4f} Joules")
        print("="*70)

class PerformanceAnalyzer:
    """Calculates performance metrics based on hydrodynamic and PTO data."""
    def __init__(self, hydro_results, df, fs):
        print("\n--- Initializing Performance Analyzer ---")
        self.results = hydro_results.copy(); self.df = df; self.fs = fs
        self.results['performance'] = {}
        self.has_pto_data = ('Pressure_Pa' in df.columns and 'Flow_Rate_m3s' in df.columns and df['Pressure_Pa'].abs().sum() > 1e-6 and df['Flow_Rate_m3s'].abs().sum() > 1e-9)
        if not self.has_pto_data: print("Warning: PTO data is missing or zero. Performance metrics will not be calculated.")

    def calculate_incident_wave_power_spectral(self):
        """Calculates incident wave power from gauge data using spectral analysis."""
        print("\n--- Calculating Incident Wave Power (Spectral Method) ---"); all_hs = []; all_te = []
        for gauge_col in INCIDENT_GAUGE_COLUMNS:
            if gauge_col not in self.df.columns: print(f"  - Warning: Incident gauge '{gauge_col}' not found. Skipping."); continue
            eta = self.df[gauge_col].values; freqs, S = welch(eta, fs=self.fs, nperseg=len(eta)); df_freq = freqs[1] - freqs[0]
            m0 = np.sum(S * df_freq); m_minus_1 = np.sum(S[1:] / freqs[1:] * df_freq) if m0 > 0 else 0
            if m0 > 0: all_hs.append(4 * np.sqrt(m0)); all_te.append(m_minus_1 / m0 if m0 > 0 else 0)
        if not all_hs: raise ValueError("No valid incident wave gauge data found to calculate wave power.")
        avg_hs = np.mean(all_hs); avg_te = np.mean(all_te)
        wave_power_per_meter = (RHO_WATER * G**2) / (64 * np.pi) * avg_hs**2 * avg_te
        self.results['performance']['model_hs_m'] = avg_hs; self.results['performance']['model_te_s'] = avg_te
        self.results['performance']['model_wave_power_W_m'] = wave_power_per_meter
        print(f"  - Avg. Significant Wave Height (Hs): {avg_hs:.4f} m\n  - Avg. Energy Period (Te): {avg_te:.3f} s\n  - Incident Wave Power (Model): {wave_power_per_meter:.2f} W/m")
        return wave_power_per_meter

    def calculate_hydraulic_performance(self):
        if not self.has_pto_data: return 0
        print("\n--- Calculating Hydraulic Performance ---"); self.df['Hydraulic_Power_W'] = self.df['Pressure_Pa'] * self.df['Flow_Rate_m3s']
        mean_power = self.df['Hydraulic_Power_W'].mean(); self.results['performance']['model_mean_power_W'] = mean_power
        print(f"  - Mean Absorbed Hydraulic Power (Model): {mean_power:.2f} W"); return mean_power

    def calculate_efficiency_metrics(self):
        if not self.has_pto_data: return
        print("\n--- Calculating Efficiency Metrics (CWR & Efficiency) ---")
        p_abs = self.results['performance'].get('model_mean_power_W'); p_wave = self.results['performance'].get('model_wave_power_W_m')
        if p_abs is None or p_wave is None: print("  - Prerequisite data missing. Skipping."); return
        cwr = p_abs / p_wave if p_wave > 0 else 0; self.results['performance']['model_cwr_m'] = cwr
        incident_power_on_device = p_wave * WEC_ARRAY_WIDTH
        efficiency = (p_abs / incident_power_on_device) * 100 if incident_power_on_device > 0 else 0
        self.results['performance']['model_efficiency_percent'] = efficiency
        print(f"  - Capture Width Ratio (CWR) (Model): {cwr:.3f} m\n  - Overall Efficiency (Model): {efficiency:.2f} %")