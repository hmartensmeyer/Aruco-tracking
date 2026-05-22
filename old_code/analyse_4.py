# ==============================================================================
#      *** 1. NECESSARY IMPORTS ***
# ==============================================================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.fft import fft, fftfreq
import os
from datetime import datetime
try:
    import tkinterdnd2
except ImportError:
    tkinterdnd2 = None # Make tkinterdnd2 optional

# GUI Imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading


# ==============================================================================
#      *** 2. PHYSICAL CONSTANTS & CONFIGURATION ***
# ==============================================================================
RHO_WATER = 1000  # Density of water (kg/m^3)
G = 9.81          # Acceleration due to gravity (m/s^2)
DEFAULT_EPSILON = 1e-9 # Epsilon for monotonic unwrap

# --- SCALING & EFFICIENCY PARAMETERS ---
SCALE_FACTOR = 40.0              # The geometric scale of the model (e.g., 1/40th scale -> 40.0)
INCIDENT_GAUGE_COLUMNS = ['Gauge_1', 'Gauge_2', 'Gauge_3', 'Gauge_4'] # Gauges measuring the undisturbed wave

# --- USER-DEFINED PARAMETERS (HYDRODYNAMICS) ---
WEC_ARRAY_WIDTH = 0.25       # Overall width of the buoy array (meters)
BUOY_DIAMETER = WEC_ARRAY_WIDTH # Assuming array width is buoy diameter for accordion calc
SINGLE_BUOY_MASS = 0.5       # kg
REFERENCE_BUOY_ID = 1      # Buoy ID used as the zero-phase reference
INCIDENT_WAVE_GAUGE = 'Gauge_1' # Gauge used to define incident wave properties (for phase analysis)

# --- USER-DEFINED PARAMETERS (PTO PERFORMANCE) ---
PADDLE_TRIGGER_ON_VOLTAGE = -4.0

# ==============================================================================
#      *** 3. DATA IMPORT & SETUP FUNCTIONS ***
# ==============================================================================

class ExperimentalSetup:
    """
    A data structure to store the physical layout of the experimental setup
    and calculate distances between points.
    """
    def __init__(self, positions_dict):
        # *** IMPROVEMENT ***: Standardize keys to lowercase for robustness
        self.positions = {str(name).lower(): np.array(coords) for name, coords in positions_dict.items()}
        print("ExperimentalSetup initialized with the following points:", list(self.positions.keys()))

    def get_position(self, point_name):
        # *** IMPROVEMENT ***: Standardize lookup to lowercase
        lookup_key = str(point_name).lower()
        if lookup_key not in self.positions:
            raise KeyError(f"Error: Point '{point_name}' (as '{lookup_key}') not found in the setup.")
        return self.positions[lookup_key]

def parse_gauge_metadata(gauges_filepath):
    """Parses the header of the gauge file to extract metadata."""
    metadata = {}
    with open(gauges_filepath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:10]):
            if 'Freq=' in line:
                match = re.search(r'Freq=([0-9.]+)', line)
                if match:
                    metadata['Frequency_Hz'] = float(match.group(1))
            if 'Amp=' in line:
                match = re.search(r'Amp=([0-9.]+)', line)
                if match:
                    metadata['Amplitude_m'] = float(match.group(1))
            if 'Rate (Hz)' in line:
                if i + 1 < len(lines):
                    next_line_parts = lines[i+1].split()
                    if next_line_parts:
                        try:
                            metadata['Sample_Rate_Hz'] = float(next_line_parts[0])
                        except ValueError:
                            print(f"Warning: Could not parse sample rate from line: {lines[i+1]}")
    if 'Sample_Rate_Hz' not in metadata:
        print("Warning: Sample rate not found in header. Inferring from first two timestamps...")
        try:
            # *** IMPROVEMENT ***: Use more rows for a more stable inference
            temp_df = pd.read_csv(gauges_filepath, sep='\t', skiprows=6, header=None, nrows=10)
            time_diff = np.mean(np.diff(temp_df.iloc[:, 0]))
            if time_diff > 0:
                inferred_rate = 1.0 / time_diff
                metadata['Sample_Rate_Hz'] = inferred_rate
                print(f"Inferred sample rate: {inferred_rate:.2f} Hz")
        except Exception as e:
            print(f"Could not infer sample rate. Error: {e}")
    return metadata

def import_pressure_flow_data(filepath):
    """
    Corrected to import and process the pressure/flow TXT file by reading
    columns by their fixed integer position, which is robust for this format.
    """
    print(f"\n--- Importing Pressure/Flow data from: {filepath} ---")
    if not filepath or not os.path.exists(filepath):
        messagebox.showerror("File Not Found", f"The file was not found at the specified path:\n{filepath}")
        return pd.DataFrame()

    try:
        df_raw = pd.read_csv(filepath, sep='\t', header=0, engine='python')
        df = pd.DataFrame({
            # *** CORRECTION ***: Use the actual time data from the first column, not the index.
            # This is crucial for correct synchronization.
            'Time_PF_Rel_s':   df_raw.index,
            'TriggerPaddle_V': df_raw.iloc[:, 0],
            'Flow_lpm':        df_raw.iloc[:, 3],
            'Pressure_bar':    df_raw.iloc[:, 4]
        })
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        print("Successfully imported data using fixed column positions (0, 1, 4, 5).")
        df['Pressure_Pa'] = df['Pressure_bar'] * 100000
        df['Flow_Rate_m3s'] = df['Flow_lpm'] / 60000
        df['Sync_Signal_PF'] = df['TriggerPaddle_V'] < PADDLE_TRIGGER_ON_VOLTAGE

        if df['Sync_Signal_PF'].sum() == 0:
            print("\nWARNING: No 'ON' signals were found in the TriggerPaddle column after processing.")

        return df
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("File Read Error", f"An unhandled error occurred while processing the Pressure/Flow file.\n\nError: {e}")
        return pd.DataFrame()

def synchronize_wec_data(dof_filepath, gauges_filepath):
    """
    Imports and synchronizes multi-buoy 6-DOF and wave gauge data.
    Returns the synchronized DataFrame, metadata, AND the absolute start time of the sync window.
    """
    print("\n--- Starting Synchronization Process (DOF & Gauges) ---")
    dof_df = pd.read_csv(dof_filepath)
    if 'LED_On' not in dof_df.columns:
         raise KeyError("Critical Error: 'LED_On' column not found in the DOF data. Cannot determine sync window.")

    sync_window = dof_df[dof_df['LED_On'] > 0.5]
    if sync_window.empty:
        raise ValueError("No synchronization window found! 'LED_On' column has no active signal.")

    start_time = sync_window['Time_Rel_Sec'].min()
    end_time = sync_window['Time_Rel_Sec'].max()
    print(f"Master Sync window found: Start={start_time:.3f}s, End={end_time:.3f}s")

    dof_df = dof_df[(dof_df['Time_Rel_Sec'] >= start_time) & (dof_df['Time_Rel_Sec'] <= end_time)].copy()

    metadata = parse_gauge_metadata(gauges_filepath)
    gauges_df = pd.read_csv(gauges_filepath, sep='\t', skiprows=6, header=None)
    gauge_cols = [f'Gauge_{i}' for i in range(1, gauges_df.shape[1])]
    gauges_df.columns = ['Time_Gauge_Rel'] + gauge_cols
    gauges_df['Time_Rel_Sec'] = gauges_df['Time_Gauge_Rel']
    gauges_df = gauges_df[(gauges_df['Time_Rel_Sec'] >= start_time) & (gauges_df['Time_Rel_Sec'] <= end_time)]

    print("Resampling gauge data to match DOF timestamps...")
    # *** IMPROVEMENT ***: Use pd.to_numeric to avoid potential dtype issues with time
    target_index = pd.to_timedelta(np.sort(pd.to_numeric(dof_df['Time_Rel_Sec'], errors='coerce').unique()), unit='s')
    gauges_df.index = pd.to_timedelta(pd.to_numeric(gauges_df['Time_Rel_Sec'], errors='coerce'), unit='s')
    # Use 'nearest' and then interpolate to handle time gaps better
    gauges_resampled = gauges_df[gauge_cols].reindex(target_index, method='nearest', limit=1).interpolate()
    gauges_resampled['Time_Rel_Sec'] = gauges_resampled.index.total_seconds()

    final_df = pd.merge(dof_df, gauges_resampled, on='Time_Rel_Sec', how='left')
    final_df = final_df.rename(columns={'Time_Rel_Sec': 'Time_s'})
    final_df.dropna(subset=['Gauge_1'], inplace=True)
    return final_df, metadata, start_time

def synchronize_all_data(dof_filepath, gauges_filepath, pto_filepath):
    """
    Orchestrates the full synchronization of DOF, Gauges, and the new Pressure/Flow data.
    """
    print("\n" + "="*50 + "\n--- STARTING FULL DATA SYNCHRONIZATION ---\n" + "="*50)
    dof_gauge_df, metadata, dof_sync_start_time = synchronize_wec_data(dof_filepath, gauges_filepath)
    pto_df = import_pressure_flow_data(pto_filepath)

    if not pto_df.empty:
        print("\n--- Aligning Pressure/Flow Timeline ---")
        pto_trigger_window = pto_df[pto_df['Sync_Signal_PF'] == True]
        if pto_trigger_window.empty:
            raise ValueError("No paddle trigger 'on' signal found in the Pressure/Flow data file. Cannot synchronize.")

        # This logic is now robust thanks to the corrected time column in pto_df
        pto_relative_start_time = pto_trigger_window['Time_PF_Rel_s'].min()
        pto_df['Time_s'] = (pto_df['Time_PF_Rel_s'] - pto_relative_start_time) + dof_sync_start_time

        print("Resampling PTO data to merge with DOF/Gauge data...")
        pto_df.set_index(pd.to_timedelta(pto_df['Time_s'], unit='s'), inplace=True)
        main_target_index = pd.to_timedelta(dof_gauge_df['Time_s'].unique(), unit='s')

        pto_cols_to_resample = ['Pressure_Pa', 'Flow_Rate_m3s']
        # Use 'nearest' then interpolate for better results
        pto_resampled = pto_df[pto_cols_to_resample].reindex(main_target_index, method='nearest', limit=1).interpolate()
        pto_resampled['Time_s'] = pto_resampled.index.total_seconds()

        final_df = pd.merge(dof_gauge_df, pto_resampled, on='Time_s', how='left')
        final_df[pto_cols_to_resample] = final_df[pto_cols_to_resample].fillna(method='ffill').fillna(method='bfill')
    else:
        final_df = dof_gauge_df
        final_df['Pressure_Pa'] = 0
        final_df['Flow_Rate_m3s'] = 0
        print("No PTO data was loaded. Proceeding with DOF/Gauge data only.")

    print(f"\nFinal synchronized DataFrame created. Shape: {final_df.shape}\n--- FULL SYNCHRONIZATION COMPLETE! ---")
    return final_df, metadata

# ==============================================================================
#      *** 4. ANALYSIS & REPORTING CLASSES ***
# ==============================================================================

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

        # *** CORRECTION ***: Loop through gauges and gracefully skip any not in the setup
        for gauge_col in selected_gauges:
            try:
                # Get position from setup (keys are standardized to lowercase)
                location = self.setup.get_position(gauge_col)[0] # We only need X-position

                # If position was found, analyze this gauge
                yf_gauge = fft(df_single_buoy_view[gauge_col].to_numpy())
                wave_phases.append(np.angle(yf_gauge[max_idx]))
                wave_amplitudes.append(2.0 / N * np.abs(yf_gauge[max_idx]))
                gauge_locations.append(location)

            except KeyError:
                # This is the fix: If a gauge from data is not in the setup,
                # print a warning and safely skip it instead of crashing.
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
            # Coordinate System: Heave is Y-axis motion
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
            # 1. Get buoy's motion phase (Heave = Y-axis)
            heave_signal = buoy_df['CoG_Pos_Y_Rel'].to_numpy()
            phase_buoy_motion = np.angle(fft(heave_signal)[max_idx])

            # 2. Calculate "expected" wave phase at buoy's location (Surge = X-axis)
            buoy_pos_x = buoy_df['CoG_Pos_X_Rel'].mean()
            distance_from_gauge = buoy_pos_x - gauge_pos_x
            phase_wave_at_buoy = phase_at_gauge - k * distance_from_gauge

            # 3. Calculate the difference (motion phase - wave phase)
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
        
        # Define cutoff frequency for high-pass filter to remove drift
        cutoff_frequency = dominant_freq / 5.0
        print(f"Applying high-pass filter with cutoff at {cutoff_frequency:.3f} Hz to remove drift.")

        # *** IMPROVEMENT ***: Standardize coordinate system with clear labels
        motion_types = {
            'Heave': 'CoG_Pos_Y_Rel', # Vertical
            'Surge': 'CoG_Pos_X_Rel', # Along wave direction
            'Sway':  'CoG_Pos_Z_Rel'  # Transverse
        }

        for buoy_id in self.buoy_ids:
            buoy_df = self.df[self.df['Buoy_ID'] == buoy_id]
            raos[buoy_id] = {}
            for motion_name, col_name in motion_types.items():
                if col_name not in buoy_df.columns: continue
                motion_signal = buoy_df[col_name].to_numpy()
                filtered_motion_signal = self._highpass_filter(motion_signal, cutoff_frequency, self.fs)
                # Amplitude for a sinusoidal-like signal from its filtered time series
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
            # *** CORRECTION ***: Ensure column names match standardized coordinate system
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
        """
        *** IMPROVEMENT ***: Calculates the GAP oscillation between adjacent buoys.
        A negative gap indicates collision.
        """
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

            # Coordinate System: Surge is X-axis motion
            surge_1 = self.df[self.df['Buoy_ID'] == buoy_id_1]['CoG_Pos_X_Rel'].to_numpy()
            surge_2 = self.df[self.df['Buoy_ID'] == buoy_id_2]['CoG_Pos_X_Rel'].to_numpy()
            min_len = min(len(surge_1), len(surge_2))
            surge_1, surge_2 = surge_1[:min_len], surge_2[:min_len]

            center_to_center_distance = surge_2 - surge_1
            # Gap = (Center-to-Center Distance) - (Radius_1 + Radius_2), which is C-C_dist - Diameter
            relative_gap = center_to_center_distance - BUOY_DIAMETER

            mean_gap = np.mean(relative_gap)
            min_gap = np.min(relative_gap)
            max_gap = np.max(relative_gap)

            # Frequency analysis on the C-C distance oscillation (same as gap oscillation)
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

    # *** NEW METHOD ***
    def plot_drift_removal_diagnostic(self, buoy_id, motion_type='Heave'):
        """
        Plots a comparison of the original, filtered, and drift signals
        to visually verify the effectiveness of the high-pass filter.
        """
        if 'raos' not in self.results:
            print("Error: Run calculate_raos() first to define filter parameters.")
            return

        motion_map = {'Heave': 'CoG_Pos_Y_Rel', 'Surge': 'CoG_Pos_X_Rel', 'Sway': 'CoG_Pos_Z_Rel'}
        col_name = motion_map.get(motion_type)
        if not col_name:
            print(f"Error: Motion type '{motion_type}' not recognized.")
            return

        buoy_df = self.df[self.df['Buoy_ID'] == buoy_id]
        if buoy_df.empty:
            print(f"Error: No data for Buoy ID {buoy_id}.")
            return

        original_signal = buoy_df[col_name].to_numpy()
        time_vector = buoy_df['Time_s'].to_numpy()
        dominant_freq = self.results['wave_properties']['frequency']
        cutoff_frequency = dominant_freq / 5.0

        filtered_signal = self._highpass_filter(original_signal, cutoff_frequency, self.fs)
        drift_signal = original_signal - filtered_signal # The part that was removed

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(time_vector, original_signal, label='1. Original Signal', color='blue', alpha=0.7)
        ax.plot(time_vector, drift_signal, label='2. Removed Drift (Low-pass)', color='red', linestyle='--', lw=2.5)
        ax.plot(time_vector, filtered_signal, label='3. Filtered Signal (High-pass)', color='green', alpha=0.9)
        ax.set_title(f'Drift Removal Diagnostic for Buoy {buoy_id} - {motion_type} Motion', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Position (m)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    def plot_accordion_effect(self):
        """
        *** IMPROVEMENT ***: Generates plots to visualize the surge GAP oscillation.
        """
        if 'accordion_effect' not in self.results or not self.results['accordion_effect']:
            print("No accordion effect data to plot. Run analyze_surge_accordion_effect() first.")
            return

        print("\n--- Generating Accordion Effect (Gap) Diagnostic Plots ---")
        time_vector = self.df['Time_s'].unique()

        for pair_name, data in self.results['accordion_effect'].items():
            gap_signal = data['time_series_gap_data']
            min_len = min(len(time_vector), len(gap_signal))

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=False,
                                           gridspec_kw={'height_ratios': [2, 1]})
            fig.suptitle(f'Gap Analysis for Buoy Pair: {pair_name}', fontsize=18)

            ax1.plot(time_vector[:min_len], gap_signal[:min_len], label='Instantaneous Gap', color='purple')
            ax1.axhline(data['mean_gap_m'], color='red', linestyle='--', label=f"Mean Gap: {data['mean_gap_m']:.3f} m")
            if data['min_gap_m'] < 0:
                ax1.axhspan(data['min_gap_m'], 0, color='red', alpha=0.2, label='Collision Occurred')
            ax1.set_title('Time-Domain View: Gap Between Buoy Surfaces')
            ax1.set_ylabel('Gap (m)'); ax1.legend(); ax1.grid(True, which='both')

            N = len(gap_signal)
            yf = fft(gap_signal - data['mean_gap_m'])
            xf = fftfreq(N, 1.0 / self.fs)
            ax2.plot(xf[:N//2], (2.0/N) * np.abs(yf[:N//2]), color='darkorange')
            ax2.axvline(data['oscillation_frequency_hz'], color='black', ls='--', label=f"Dom Freq: {data['oscillation_frequency_hz']:.3f} Hz")
            ax2.set_title('Frequency-Domain View: Gap Oscillation Spectrum')
            ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Gap Oscillation Amp (m)')
            ax2.set_xlim(0, xf[N//2 -1] if N > 2 else 1); ax2.legend(); ax2.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def summarize_results(self):
        # ... (This method remains largely the same but I've added the new results)
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
            print(f"Wave-to-Buoy Heave Phase Lag (°):")
            print(f"  " + " | ".join([f"Buoy {bid}: {p:.1f}°" for bid, p in self.results['wave_to_buoy_phase_deg'].items()]))
        print("-" * 70)
        if 'relative_heave_phase_deg' in self.results:
            print(f"Buoy-to-Buoy Relative Heave Phase (°, ref: Buoy {REFERENCE_BUOY_ID}):")
            print(f"  " + " | ".join([f"Buoy {bid}: {p:.1f}°" for bid, p in self.results['relative_heave_phase_deg'].items()]))
        print("-" * 70)
        if 'mean_drift' in self.results:
            print("Mean Drift Displacement (m):")
            for buoy_id, drift_dict in self.results['mean_drift'].items(): print(f"  Buoy {buoy_id} -> Surge: {drift_dict['Surge_Drift_m']:.4f}, Sway: {drift_dict['Sway_Drift_m']:.4f}")
        print("-" * 70)
        # *** IMPROVEMENT ***: Added accordion effect to summary
        if 'accordion_effect' in self.results:
            print("Accordion Effect (Relative Surge Gap Oscillation):")
            for pair, data in self.results['accordion_effect'].items():
                print(f"  {pair}: Mean Gap: {data['mean_gap_m']:.3f} m | Min Gap: {data['min_gap_m']:.3f} m (Collision if < 0) | Osc. Amp: {data['oscillation_amplitude_m']:.4f} m")
        print("-" * 70)
        if 'mean_system_ke' in self.results: print(f"System Energy Proxy:\n  - Average Total Kinetic Energy: {self.results['mean_system_ke']:.4f} Joules")
        print("="*70)

# The `PerformanceAnalyzer` and `ReportGenerator` classes are already excellent and require no major changes.
# I've left them as they were in your original code.
class PerformanceAnalyzer:
    """
    Calculates performance metrics based on hydrodynamic and PTO data.
    Implements spectral analysis for wave power and Froude scaling for full-scale results.
    """
    def __init__(self, hydro_results, df, fs):
        print("\n--- Initializing Performance Analyzer ---")
        self.results = hydro_results.copy() # Start with results from hydro analysis
        self.df = df
        self.fs = fs
        self.results['performance'] = {} # Create a dedicated dictionary for performance results

        # Check if PTO data is usable
        self.has_pto_data = ('Pressure_Pa' in df.columns and 'Flow_Rate_m3s' in df.columns and \
                             df['Pressure_Pa'].abs().sum() > 1e-6 and df['Flow_Rate_m3s'].abs().sum() > 1e-9)

        if not self.has_pto_data:
            print("Warning: PTO data (Pressure/Flow) is missing or zero. Performance metrics will not be calculated.")

    def calculate_incident_wave_power_spectral(self):
        """
        Calculates incident wave power from gauge data using spectral analysis.
        This is the industry-standard method for irregular (and regular) seas.
        """
        print("\n--- Calculating Incident Wave Power (Spectral Method) ---")
        all_hs = []
        all_te = []
        
        for gauge_col in INCIDENT_GAUGE_COLUMNS:
            if gauge_col not in self.df.columns:
                print(f"  - Warning: Incident gauge '{gauge_col}' not found in data. Skipping.")
                continue
            eta = self.df[gauge_col].values
            freqs, S = welch(eta, fs=self.fs, nperseg=len(eta))
            df_freq = freqs[1] - freqs[0]
            m0 = np.sum(S * df_freq)
            m_minus_1 = np.sum(S[1:] / freqs[1:] * df_freq) if m0 > 0 else 0
            if m0 > 0:
                all_hs.append(4 * np.sqrt(m0))
                all_te.append(m_minus_1 / m0 if m0 > 0 else 0)

        if not all_hs:
            raise ValueError("No valid incident wave gauge data found to calculate wave power.")

        avg_hs = np.mean(all_hs)
        avg_te = np.mean(all_te)
        wave_power_per_meter = (RHO_WATER * G**2) / (64 * np.pi) * avg_hs**2 * avg_te

        self.results['performance']['model_hs_m'] = avg_hs
        self.results['performance']['model_te_s'] = avg_te
        self.results['performance']['model_wave_power_W_m'] = wave_power_per_meter

        print(f"  - Avg. Significant Wave Height (Hs): {avg_hs:.4f} m")
        print(f"  - Avg. Energy Period (Te): {avg_te:.3f} s")
        print(f"  - Incident Wave Power (Model): {wave_power_per_meter:.2f} W/m")
        return wave_power_per_meter

    def calculate_hydraulic_performance(self):
        if not self.has_pto_data: return 0
        print("\n--- Calculating Hydraulic Performance ---")
        self.df['Hydraulic_Power_W'] = self.df['Pressure_Pa'] * self.df['Flow_Rate_m3s']
        mean_power = self.df['Hydraulic_Power_W'].mean()
        self.results['performance']['model_mean_power_W'] = mean_power
        print(f"  - Mean Absorbed Hydraulic Power (Model): {mean_power:.2f} W")
        return mean_power

    def calculate_efficiency_metrics(self):
        """Calculates CWR and conventional efficiency based on previously calculated power values."""
        if not self.has_pto_data: return
        print("\n--- Calculating Efficiency Metrics (CWR & Efficiency) ---")
        p_abs = self.results['performance'].get('model_mean_power_W')
        p_wave = self.results['performance'].get('model_wave_power_W_m')
        if p_abs is None or p_wave is None:
            print("  - Prerequisite data (absorbed or wave power) missing. Skipping.")
            return

        cwr = p_abs / p_wave if p_wave > 0 else 0
        self.results['performance']['model_cwr_m'] = cwr
        incident_power_on_device = p_wave * WEC_ARRAY_WIDTH
        efficiency = (p_abs / incident_power_on_device) * 100 if incident_power_on_device > 0 else 0
        self.results['performance']['model_efficiency_percent'] = efficiency
        print(f"  - Capture Width Ratio (CWR) (Model): {cwr:.3f} m")
        print(f"  - Overall Efficiency (Model): {efficiency:.2f} % (Based on array width of {WEC_ARRAY_WIDTH}m)")

    def scale_results_to_full_scale(self):
        """Applies Froude scaling to key performance metrics."""
        if 'performance' not in self.results: return
        print(f"\n--- Scaling Results to Full-Scale (Scale Factor: {SCALE_FACTOR}) ---")
        p_model = self.results['performance'].get('model_mean_power_W', 0)
        cwr_model = self.results['performance'].get('model_cwr_m', 0)
        hs_model = self.results['performance'].get('model_hs_m', 0)
        te_model = self.results['performance'].get('model_te_s', 0)

        p_full_kw = (p_model * SCALE_FACTOR**3.5) / 1000.0
        cwr_full = cwr_model * SCALE_FACTOR
        hs_full = hs_model * SCALE_FACTOR
        te_full = te_model * np.sqrt(SCALE_FACTOR)

        self.results['performance']['full_scale_mean_power_kW'] = p_full_kw
        self.results['performance']['full_scale_cwr_m'] = cwr_full
        self.results['performance']['full_scale_hs_m'] = hs_full
        self.results['performance']['full_scale_te_s'] = te_full
        
        print(f"  - Full-Scale Hs: {hs_full:.2f} m")
        print(f"  - Full-Scale Te: {te_full:.2f} s")
        print(f"  - Full-Scale Mean Power: {p_full_kw:.2f} kW")
        print(f"  - Full-Scale CWR: {cwr_full:.2f} m")

class ReportGenerator:
    """Generates plots and a full Markdown report of the analysis."""
    def __init__(self, analysis_results, df, report_dir='reports'):
        self.results = analysis_results
        self.df = df
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        print(f"\n--- ReportGenerator initialized. Reports will be saved to '{self.report_dir}/' ---")

    def _save_plot(self, fig, filename):
        path = os.path.join(self.report_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return os.path.basename(path)

    def plot_power_timeseries(self):
        if 'Hydraulic_Power_W' not in self.df.columns: return None
        fig, ax1 = plt.subplots(figsize=(15, 7))
        plt.style.use('seaborn-v0_8-whitegrid')
        mean_power = self.results.get('performance', {}).get('model_mean_power_W', 0)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Wave Elevation (m)', color='royalblue')
        ax1.plot(self.df['Time_s'], self.df[INCIDENT_WAVE_GAUGE], color='royalblue', label='Incident Wave')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Instantaneous Power (W)', color='firebrick')
        ax2.plot(self.df['Time_s'], self.df['Hydraulic_Power_W'], color='firebrick', alpha=0.7, label='Instantaneous Power')
        ax2.axhline(y=mean_power, color='darkred', linestyle='--', linewidth=2, label=f'Mean Power: {mean_power:.2f} W')
        ax2.tick_params(axis='y', labelcolor='firebrick')
        
        fig.suptitle('Wave Elevation vs. Absorbed Power (Model Scale)', fontsize=16)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        return self._save_plot(fig, 'power_timeseries.png')

    def generate_markdown_report(self, filename='performance_report.md'):
        print(f"--- Generating Markdown Report: {filename} ---")
        power_ts_path = self.plot_power_timeseries()
        
        # Helper to safely get nested results
        def get_res(key, default=0, precision=2):
            return f"{self.results.get('performance', {}).get(key, default):.{precision}f}"
        
        md = f"""# 🌊 WEC Performance Analysis Report
> **Date of Report:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---
## 1. Executive Summary
This report details the performance of the WEC array based on the provided experimental data. Key metrics include absorbed power and Capture Width Ratio (CWR) at both model and full scale.
### Full-Scale Performance Highlights
| Metric | Value | Unit |
|---|---|---|
| **Equivalent Wave State (Hs)** | {get_res('full_scale_hs_m')} | m |
| **Equivalent Wave State (Te)** | {get_res('full_scale_te_s')} | s |
| **Mean Absorbed Power** | **{get_res('full_scale_mean_power_kW')}** | **kW** |
| **Capture Width Ratio (CWR)** | **{get_res('full_scale_cwr_m')}** | **m** |
---
## 2. Power Absorption Analysis (Model Scale)
The following plot shows the relationship between the incoming wave and the instantaneous power absorbed by the PTO system.
"""
        if power_ts_path:
            md += f"\n![Power Time Series]({power_ts_path})\n"
        md += f"""
### Model-Scale Performance Metrics
| Metric | Value | Unit |
|---|---|---|
| Incident Wave Power | {get_res('model_wave_power_W_m')} | W/m |
| Mean Absorbed Hydraulic Power | {get_res('model_mean_power_W')} | W |
| Capture Width Ratio (CWR) | {get_res('model_cwr_m', precision=3)} | m |
| Overall Efficiency | {get_res('model_efficiency_percent')} | % |
"""
        report_path = os.path.join(self.report_dir, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ Report successfully generated at: {report_path}")
        return report_path

# ==============================================================================
#      *** 5. GRAPHICAL USER INTERFACE ***
# ==============================================================================

class AnalysisGUI(tkinterdnd2.Tk if tkinterdnd2 else tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WEC Performance Analysis Tool")
        self.geometry("700x350")
        self.dof_file = tk.StringVar()
        self.gauges_file = tk.StringVar()
        self.pto_file = tk.StringVar()
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(main_frame, text="1. Select Data Files (or Drag & Drop)", font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky='w', pady=5)
        self.create_file_selector(main_frame, "DOF File (.csv):", self.dof_file, 1)
        self.create_file_selector(main_frame, "Gauges File (.txt):", self.gauges_file, 2)
        self.create_file_selector(main_frame, "Pressure/Flow File (.txt):", self.pto_file, 3)
        self.run_button = ttk.Button(main_frame, text="Run Full Analysis", command=self.start_analysis_thread, style="Accent.TButton")
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, ipady=5)
        ttk.Style().configure("Accent.TButton", font=("Helvetica", 12, "bold"))
        self.status_var = tk.StringVar(value="Ready. Select all files and click Run.")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w', padding=5).pack(side=tk.BOTTOM, fill=tk.X)

    def create_file_selector(self, parent, label_text, string_var, row):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=string_var, width=70)
        entry.grid(row=row, column=1, sticky='ew', padx=5)
        if tkinterdnd2:
            # *** THIS IS THE CORRECTED LINE ***
            entry.drop_target_register(tkinterdnd2.DND_FILES) 
            entry.dnd_bind('<<Drop>>', lambda event: self.handle_drop(event, string_var))
        ttk.Button(parent, text="Browse...", command=lambda: self.browse_file(string_var, label_text)).grid(row=row, column=2, sticky='e', padx=5)

    def handle_drop(self, event, string_var):
        # Strip potential curly braces from the filepath if dragged from some terminals
        filepath = event.data.strip('{}')
        string_var.set(filepath)
        self.status_var.set(f"Loaded '{os.path.basename(filepath)}'.")

    def browse_file(self, string_var, title):
        filetypes = [("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(title=f"Select {title}", filetypes=filetypes)
        if filepath:
            string_var.set(filepath)
            self.status_var.set(f"Loaded '{os.path.basename(filepath)}'.")

    def start_analysis_thread(self):
        if not all([self.dof_file.get(), self.gauges_file.get(), self.pto_file.get()]):
            messagebox.showerror("Input Error", "Please select all three data files before running.")
            return
        self.run_button.config(state="disabled")
        self.status_var.set("Analysis in progress... Please wait.")
        threading.Thread(target=self.run_full_analysis, daemon=True).start()

    def run_full_analysis(self):
        try:
            dof_path = self.dof_file.get()
            report_directory = os.path.dirname(dof_path)

            self.status_var.set("Step 1/5: Synchronizing data...")
            final_df, metadata = synchronize_all_data(dof_path, self.gauges_file.get(), self.pto_file.get())

            self.status_var.set("Step 2/5: Performing hydrodynamic analysis...")
            # Define the known physical setup of the wave gauges
            gauge_x_pos = [0, 0.220, 0.3, 0.5, 0.918, 1.147, 1.370, 1.830]
            setup = ExperimentalSetup({f'gauge_{i+1}': [x,0,0] for i, x in enumerate(gauge_x_pos)})
            
            hydro_analysis = HydrodynamicAnalysis(final_df, metadata, setup, transient_duration=5.0)
            # Run all hydrodynamic analyses
            hydro_analysis.analyze_monochromatic_wave_properties()
            hydro_analysis.calculate_raos()
            hydro_analysis.calculate_wave_to_buoy_phase()
            hydro_analysis.calculate_mean_drift()
            hydro_analysis.analyze_surge_accordion_effect()
            hydro_analysis.summarize_results()

            self.status_var.set("Step 3/5: Calculating performance metrics...")
            perf_analyzer = PerformanceAnalyzer(hydro_analysis.results, hydro_analysis.df, hydro_analysis.fs)
            perf_analyzer.calculate_incident_wave_power_spectral()
            perf_analyzer.calculate_hydraulic_performance()
            perf_analyzer.calculate_efficiency_metrics()

            self.status_var.set("Step 4/5: Scaling results to full-scale...")
            perf_analyzer.scale_results_to_full_scale()

            self.status_var.set("Step 5/5: Generating report...")
            reporter = ReportGenerator(perf_analyzer.results, perf_analyzer.df, report_dir=report_directory)
            report_path = reporter.generate_markdown_report()

            self.status_var.set(f"Analysis complete! Report saved.")
            messagebox.showinfo("Success", f"Analysis finished successfully!\n\nReport saved to:\n{report_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set("Error! Analysis failed. Check console for details.")
            messagebox.showerror("Analysis Failed", f"An error occurred:\n\n{e}")
        finally:
            self.run_button.config(state="normal")
            
            
# ==============================================================================
#      *** 6. MAIN EXECUTION SCRIPT ***
# ==============================================================================
if __name__ == '__main__':
    app = AnalysisGUI()
    app.mainloop()