import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import tkinterdnd2 as dnd
import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
import threading

# <<< NOUVEAU : Importations pour la visualisation >>>
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Constants ---
PADDLE_TRIGGER_ON_VOLTAGE = -2.5
LED_TRIGGER_ON_THRESHOLD = 0.5

# =============================================================================
# DATA INSPECTOR CLASS (Unchanged)
# =============================================================================
class DataInspector:
    def __init__(self, filepath: Path, file_type: str):
        self.filepath = filepath
        self.file_type = file_type.upper()
        self.report = {'filepath': filepath, 'file_type': self.file_type, 'overall_status': 'Success', 'details': []}
        self.df = None

    def _add_detail(self, check, status, message):
        self.report['details'].append({'check': check, 'status': status, 'message': message})
        if status == 'Error': self.report['overall_status'] = 'Error'
        elif status == 'Warning' and self.report['overall_status'] != 'Error': self.report['overall_status'] = 'Warning'

    def run_analysis(self):
        if not self._check_readability(): return self.report
        self._check_required_columns()
        self._check_time_column()
        self._check_column_dtypes() # This method is now corrected
        self._check_sync_signal()
        if not self.report['details']: self._add_detail("Overall", "Success", "File appears to be valid.")
        return self.report

    def _check_readability(self):
        try:
            if self.file_type == 'GAUGE':
                start_line = self._find_gauge_data_start(self.filepath)
                self.df = pd.read_csv(self.filepath, sep='\t', header=None, skiprows=start_line, low_memory=False, nrows=1000)
            elif self.file_type == 'PTO':
                self.df = pd.read_csv(self.filepath, sep='\t', header=0, nrows=1000)
                if self.df.shape[1] >= 2:
                    self.df.rename(columns={self.df.columns[0]: 'Time', self.df.columns[1]: 'Trigger'}, inplace=True)
                else:
                    self._add_detail("File Format", "Error", "PTO file has fewer than 2 columns.")
            else: # DOF
                self.df = pd.read_csv(self.filepath, nrows=1000)

            if self.df.shape[0] < 2:
                self._add_detail("File Format", "Error", "Only one row was read. The file may be missing newlines between data entries.")
                return False

            self._add_detail("Readability", "Success", "File is readable and has multiple rows.")
            return True
        except Exception as e:
            self._add_detail("Readability", "Error", f"Could not read file. Error: {e}")
            return False

    def _find_gauge_data_start(self, filepath):
        try:
            with open(filepath, 'r', errors='ignore') as f:
                for i, line in enumerate(f):
                    if re.match(r'^Time\s+', line) and '\t' in line:
                        return i + 2
                    if i > 50: break
        except Exception:
            pass
        self._add_detail("Gauge Header", "Warning", "Could not reliably detect header. Assuming data starts on line 6.")
        return 6

    def _check_required_columns(self):
        required = {'DOF': ['Time_Rel_Sec', 'LED_On', 'Buoy_ID'], 'PTO': ['Time', 'Trigger'], 'GAUGE': [0]}
        if self.file_type in required:
            missing = [col for col in required[self.file_type] if col not in self.df.columns]
            if missing:
                if 'Buoy_ID' in missing and self.file_type == 'DOF':
                    self._add_detail("Required Columns", "Warning", "Missing 'Buoy_ID' column. Aggregation will not be per-buoy.")
                else:
                    self._add_detail("Required Columns", "Error", f"Missing essential columns: {', '.join(map(str, missing))}")

    def _check_time_column(self):
        time_col_name = {'DOF': 'Time_Rel_Sec', 'GAUGE': 0, 'PTO': 'Time'}.get(self.file_type)
        if time_col_name not in self.df.columns: return
        time_series = pd.to_numeric(self.df[time_col_name], errors='coerce')
        if time_series.isnull().any():
            self._add_detail("Time Column", "Error", f"{time_series.isnull().sum()} non-numeric values found.")
        time_series.dropna(inplace=True)
        if not time_series.is_monotonic_increasing:
            self._add_detail("Time Column", "Warning", "Timestamps are not sorted. Will be fixed automatically.")
        if time_series.duplicated().any():
            msg = f"Found {time_series.duplicated().sum()} duplicate timestamps."
            if 'Buoy_ID' in self.df.columns:
                msg += " Will be averaged automatically per Buoy_ID."
            else:
                msg += " Will be averaged automatically."
            self._add_detail("Time Column", "Warning", msg)

    def _check_column_dtypes(self):
        """
        CORRECTED: Checks all data columns for non-numeric values.
        The original version incorrectly skipped columns with mixed data types.
        """
        for col in self.df.columns:
            # The main time column is checked separately in _check_time_column
            if col in ['Time_Rel_Sec', 0, 'Time']:
                continue

            series = self.df[col]

            # Skip columns that are entirely empty
            if series.isnull().all():
                continue

            # The original number of missing values
            original_nulls = series.isnull().sum()
            # Coerce to numeric, turning non-numeric strings into NaN (Not a Number)
            numeric_series = pd.to_numeric(series, errors='coerce')
            # The new number of missing values
            coerced_nulls = numeric_series.isnull().sum()

            # If the number of nulls increased, it means non-numeric values were found
            if coerced_nulls > original_nulls:
                bad_rows = coerced_nulls - original_nulls
                self._add_detail("Data Types", "Warning", f"Column '{col}' contains {bad_rows} non-numeric value(s) that will be ignored.")


    def _check_sync_signal(self):
        try:
            if self.file_type == 'DOF':
                if (self.df['LED_On'] > LED_TRIGGER_ON_THRESHOLD).sum() == 0:
                    self._add_detail("Sync Signal", "Error", "'LED_On' signal is never active.")
            elif self.file_type == 'PTO':
                if (self.df['Trigger'] < PADDLE_TRIGGER_ON_VOLTAGE).sum() == 0:
                    self._add_detail("Sync Signal", "Warning", "'Trigger' signal is never active; PTO alignment will fail.")
        except KeyError: pass

# =============================================================================
# BACKEND DATA PROCESSING CLASS (GAUGE as Master + Final Column Formatting)
# =============================================================================
class DataSynchronizer:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.dof_df = None
        self.gauge_df = None
        self.pto_df = None
        self.metadata = {}
        # Keep track of column names from each source file
        self.original_dof_columns = []
        self.original_gauge_columns = []
        self.original_pto_columns = []

    def _log(self, message):
        if self.verbose: logging.info(message)

    def _find_gauge_data_start(self, filepath):
        try:
            with open(filepath, 'r', errors='ignore') as f:
                for i, line in enumerate(f):
                    if re.match(r'^Time\s+', line) and '\t' in line:
                        self._log(f"Detected gauge data header on line {i+1}. Data should start on line {i+3}.")
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
        
        buoy_id_col = next((c for c in df.columns if c.lower() == 'buoy_id'), None)
        if buoy_id_col and buoy_id_col != 'Buoy_id':
            df.rename(columns={buoy_id_col: 'Buoy_id'}, inplace=True)
        
        grouping_cols = ['Time_s']
        if 'Buoy_id' in df.columns:
            grouping_cols.append('Buoy_id')

        if df.duplicated(subset=grouping_cols).any():
            self._log(f"Handling duplicates by aggregating. Grouping by: {', '.join(grouping_cols)}")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            object_cols = df.select_dtypes(include='object').columns.tolist()
            agg_dict = {col: 'mean' for col in numeric_cols if col not in grouping_cols}
            agg_dict.update({col: 'first' for col in object_cols if col not in grouping_cols})
            df = df.groupby(grouping_cols, as_index=False).agg(agg_dict)
        
        return df.sort_values(grouping_cols).reset_index(drop=True)

    def load_dof_data(self, filepath: Path):
        self._log(f"--- Loading DOF Data from: {filepath.name} ---")
        try:
            df = pd.read_csv(filepath)
            self.original_dof_columns = [c for c in df.columns if c != 'Time_Rel_Sec']
            df = self._clean_and_uniquify_time(df, time_col='Time_Rel_Sec')
            self.dof_df = df
            self._log(f"Successfully loaded DOF data.")
        except Exception as e:
            logging.error(f"Failed to process DOF file {filepath.name}. Error: {e}", exc_info=True)

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
            self._log(f"Successfully loaded Gauge data.")
        except Exception as e: 
            logging.error(f"Failed to process gauge file {filepath.name}. Error: {e}", exc_info=True)

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
            self._log(f"Successfully loaded PTO data.")
        except Exception as e: 
            logging.error(f"Failed to process PTO file {filepath.name}. Error: {e}", exc_info=True)

    def get_sync_visualization_data(self):
        # This method remains unchanged
        self._log("--- Preparing data for synchronization visualization ---")
        if self.dof_df is None or self.dof_df.empty or self.pto_df is None or self.pto_df.empty: return None
        dof_sync_window = self.dof_df[self.dof_df['LED_On'] > LED_TRIGGER_ON_THRESHOLD]
        if dof_sync_window.empty: return None
        dof_start_time = dof_sync_window['Time_s'].min()
        pto_sync_window = self.pto_df[self.pto_df['Sync_Signal_PF'] == True]
        if pto_sync_window.empty: return None
        pto_start_time = pto_sync_window['Time_s'].min()
        time_offset = pto_start_time - dof_start_time 
        pto_time_aligned = self.pto_df['Time_s'] - time_offset
        return {
            'dof_time': self.dof_df['Time_s'], 'dof_signal': self.dof_df['LED_On'],
            'pto_time': self.pto_df['Time_s'], 'pto_signal': self.pto_df['TriggerPaddle_V'],
            'pto_time_aligned': pto_time_aligned, 'dof_start': dof_start_time, 'pto_start_raw': pto_start_time
        }

    def synchronize(self):
        self._log("\n" + "="*50 + "\n--- STARTING FULL DATA SYNCHRONIZATION (GAUGE as Master) ---\n" + "="*50)
        
        if self.gauge_df is None or self.gauge_df.empty:
            logging.error("GAUGE file is required for synchronization but is not loaded.")
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
                logging.warning("No sync signal ('LED_On') found in DOF file. It cannot be aligned and will be excluded.")
                dof_aligned = None

        if pto_aligned is not None:
            sync_window = pto_aligned[pto_aligned['Sync_Signal_PF'] == True]
            if not sync_window.empty:
                pto_sync_time = sync_window['Time_s'].min()
                offset = pto_sync_time - master_start_time
                pto_aligned['Time_s'] -= offset
                self._log(f"PTO sync event found at {pto_sync_time:.4f}s. Shifting PTO timeline by {-offset:.4f}s.")
            else:
                logging.warning("No sync signal ('Trigger') found in PTO file. It will be excluded.")
                pto_aligned = None
        
        self._log("Merging all aligned data streams...")
        final_df = self.gauge_df.copy()
        if dof_aligned is not None:
            final_df = pd.merge(final_df, dof_aligned, on='Time_s', how='outer', sort=True)
        if pto_aligned is not None:
            final_df = pd.merge(final_df, pto_aligned, on='Time_s', how='outer', sort=True)

        self._log("Performing final interpolation...")
        if 'Buoy_id' in final_df.columns:
            final_df['Buoy_id'] = final_df['Buoy_id'].ffill().bfill()

        dof_cols = [c for c in self.original_dof_columns if c in final_df.columns and c != 'Buoy_id']
        gauge_cols = [c for c in self.original_gauge_columns if c in final_df.columns]
        pto_cols = [c for c in self.original_pto_columns if c in final_df.columns]

        if 'Buoy_id' in final_df.columns and dof_cols:
            self._log(f"Interpolating {len(dof_cols)} DOF columns per buoy...")
            final_df[dof_cols] = final_df.groupby('Buoy_id')[dof_cols].transform(
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
        
        final_df.dropna(subset=['Buoy_id'] if 'Buoy_id' in final_df.columns else ['Time_s'], inplace=True)
        if 'Buoy_id' in final_df.columns:
            final_df['Buoy_id'] = pd.to_numeric(final_df['Buoy_id'], errors='coerce').round().astype('Int64')

        # <<< NEW SECTION FOR FINAL COLUMN SELECTION AND REORDERING >>>
        self._log("Selecting and reordering final columns...")
        
        # Define the desired columns in the correct order
        desired_columns = [
            'Time_s', 'Buoy_id',
            'Gauge_1', 'Gauge_2', 'Gauge_3', 'Gauge_4', 'Gauge_5', 'Gauge_6', 'Gauge_7', 'Gauge_8',
            'CoG_Pos_X_Rel', 'CoG_Pos_Y_Rel', 'CoG_Pos_Z_Rel',
            'CoG_Vel_X_Rel', 'CoG_Vel_Y_Rel', 'CoG_Vel_Z_Rel',
            'CoG_Acc_X_Rel', 'CoG_Acc_Y_Rel', 'CoG_Acc_Z_Rel',
            'CoG_Quat_X_Rel', 'CoG_Quat_Y_Rel', 'CoG_Quat_Z_Rel', 'CoG_Quat_W_Rel',
            'Pressure_bar', 'Pressure_Pa', 'Flow_lpm', 'Flow_Rate_m3s'
        ]

        # Find which of the desired columns actually exist in the final dataframe
        # This handles cases where optional files (like PTO) were not provided
        existing_desired_columns = [col for col in desired_columns if col in final_df.columns]

        # Filter the dataframe to keep only those columns, in that order
        final_df = final_df[existing_desired_columns]
        
        self._log(f"--- FINAL FORMATTING COMPLETE! Final Shape: {final_df.shape} ---")
        return final_df, self.metadata
      
# =============================================================================
# FRONTEND TKINTER GUI APPLICATION (Unchanged)
# =============================================================================
class TextHandler(logging.Handler):
    # ... This class is unchanged ...
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget
    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append)
        
class SyncApp(dnd.TkinterDnD.Tk):
    # ... This class is unchanged ...
    def __init__(self):
        super().__init__()
        self.title("Data Synchronization & Validation Utility")
        self.geometry("950x700") 
        self.dof_path, self.gauge_path, self.pto_path = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.report_window = None
        self.plot_window = None 
        self._create_widgets()
        self._configure_logging()

    def _create_widgets(self):
        main_frame = tk.Frame(self, padx=10, pady=10); main_frame.pack(fill=tk.BOTH, expand=True)
        drop_frame = tk.Frame(main_frame); drop_frame.pack(fill=tk.X, pady=5)
        self._create_drop_zone(drop_frame, "DOF File (Required)", self.dof_path, self._on_drop_dof, 0)
        self._create_drop_zone(drop_frame, "Gauge File (Optional)", self.gauge_path, self._on_drop_gauge, 1)
        self._create_drop_zone(drop_frame, "PTO File (Optional)", self.pto_path, self._on_drop_pto, 2)
        
        control_frame = tk.Frame(main_frame); control_frame.pack(fill=tk.X, pady=10)
        self.analyze_button = tk.Button(control_frame, text="1. Analyze Files", command=self._run_analysis_thread); self.analyze_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.visualize_button = tk.Button(control_frame, text="2. Visualize Sync Signals", command=self._run_visualization_thread, state='disabled'); self.visualize_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.sync_button = tk.Button(control_frame, text="3. Synchronize and Save", command=self._run_sync_thread, state='disabled'); self.sync_button.pack(side=tk.LEFT, padx=5, pady=5)
        clear_button = tk.Button(control_frame, text="Clear All", command=self._clear_paths); clear_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        log_frame = tk.LabelFrame(main_frame, text="Log", padx=5, pady=5); log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', wrap=tk.WORD, height=15); self.log_text.pack(fill=tk.BOTH, expand=True)

    def _create_drop_zone(self, parent, text, var, drop_cmd, row):
        tk.Label(parent, text=f"{text}:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        label = tk.Label(parent, text="Drag & Drop File Here", relief="sunken", bg="white", width=80); label.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        label.drop_target_register(dnd.DND_FILES); label.dnd_bind('<<Drop>>', drop_cmd)
        var.trace_add("write", lambda *a, label=label, var=var: self._path_updated(label, var))
        parent.grid_columnconfigure(1, weight=1)

    def _path_updated(self, label, var):
        path = var.get()
        label.config(text=Path(path).name if path else "Drag & Drop File Here")
        self.sync_button.config(state='disabled')
        self.visualize_button.config(state='disabled')

    def _clear_paths(self):
        self.dof_path.set(""); self.gauge_path.set(""); self.pto_path.set("")
        self.log_text.configure(state='normal'); self.log_text.delete(1.0, tk.END); self.log_text.configure(state='disabled')
        if self.report_window: self.report_window.destroy()
        if self.plot_window: self.plot_window.destroy() 
        logging.info("Cleared all file paths and logs."); 
        self.sync_button.config(state='disabled')
        self.visualize_button.config(state='disabled') 

    def _handle_drop(self, event, path_var): self.after(10, lambda: path_var.set(event.data.strip('{}')))
    def _on_drop_dof(self, event): self._handle_drop(event, self.dof_path)
    def _on_drop_gauge(self, event): self._handle_drop(event, self.gauge_path)
    def _on_drop_pto(self, event): self._handle_drop(event, self.pto_path)
    def _configure_logging(self):
        log_handler = TextHandler(self.log_text); log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
        logging.getLogger().addHandler(log_handler); logging.getLogger().setLevel(logging.INFO)
    
    def _run_analysis_thread(self):
        if not self.dof_path.get(): messagebox.showerror("Error", "DOF File is required for analysis."); return
        self.analyze_button.config(state='disabled'); threading.Thread(target=self._run_file_analysis, daemon=True).start()
    
    def _run_file_analysis(self):
        logging.info("--- Starting File Analysis ---")
        reports = [DataInspector(Path(p.get()), t).run_analysis() for t, p in {'DOF': self.dof_path, 'GAUGE': self.gauge_path, 'PTO': self.pto_path}.items() if p.get()]
        logging.info("--- File Analysis Complete ---")
        self.after(0, self._display_analysis_report, reports)
        self.analyze_button.config(state='normal')

    def _display_analysis_report(self, reports):
        if self.report_window: self.report_window.destroy()
        self.report_window = tk.Toplevel(self); self.report_window.title("File Analysis Report"); self.report_window.geometry("800x400")
        tree = ttk.Treeview(self.report_window, columns=("File", "Check", "Status", "Details"), show="headings"); tree.pack(fill="both", expand=True, padx=10, pady=10)
        for col in ("File", "Check", "Status", "Details"): tree.heading(col, text=col)
        tree.column("File", width=150); tree.column("Check", width=100); tree.column("Status", width=80, anchor='center'); tree.column("Details", width=450)
        tree.tag_configure('Error', background='#ffdddd'); tree.tag_configure('Warning', background='#fffbdd'); tree.tag_configure('Success', background='#ddffdd')
        
        can_synchronize = all(r['overall_status'] != 'Error' for r in reports)
        for report in reports:
            for detail in report['details']:
                tree.insert("", "end", values=(report['filepath'].name, detail['check'], detail['status'], detail['message']), tags=(detail['status'],))
        
        if can_synchronize:
            self.sync_button.config(state='normal')
            if self.pto_path.get():
                self.visualize_button.config(state='normal')
            logging.info("Analysis complete: All critical checks passed. Ready to visualize or synchronize.")
        else:
            self.sync_button.config(state='disabled')
            self.visualize_button.config(state='disabled')
            logging.error("Analysis complete: Critical errors found. Please fix files.")

    def _run_visualization_thread(self):
        self.visualize_button.config(state='disabled')
        self.analyze_button.config(state='disabled')
        threading.Thread(target=self._prepare_and_plot_sync, daemon=True).start()

    def _prepare_and_plot_sync(self):
        try:
            visualizer = DataSynchronizer(verbose=True) 
            visualizer.load_dof_data(Path(self.dof_path.get()))
            if self.pto_path.get():
                visualizer.load_pto_data(Path(self.pto_path.get()))
            
            plot_data = visualizer.get_sync_visualization_data()
            
            if plot_data:
                self.after(0, self._display_sync_plot, plot_data)
            else:
                logging.error("Could not generate data for visualization.")

        except Exception as e:
            logging.error(f"An error occurred during visualization prep: {e}", exc_info=True)
        finally:
            self.visualize_button.config(state='normal')
            self.analyze_button.config(state='normal')
    
    def _display_sync_plot(self, plot_data):
        if self.plot_window:
            self.plot_window.destroy()

        self.plot_window = tk.Toplevel(self)
        self.plot_window.title("Synchronization Signal Alignment")
        self.plot_window.geometry("1000x800")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False, tight_layout=True)
        fig.suptitle('Verification of Time Synchronization', fontsize=16)

        ax1.set_title('Before Alignment (Raw Timelines)')
        ax1.plot(plot_data['dof_time'], plot_data['dof_signal'], label=f'DOF Signal (LED_On)', color='blue', alpha=0.8)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(plot_data['pto_time'], plot_data['pto_signal'], label=f'PTO Signal (TriggerPaddle_V)', color='red', linestyle='--', alpha=0.8)
        ax1.axvline(plot_data['dof_start'], color='blue', linestyle=':', lw=2, label=f'DOF Sync Start ({plot_data["dof_start"]:.2f}s)')
        ax1_twin.axvline(plot_data['pto_start_raw'], color='red', linestyle=':', lw=2, label=f'PTO Sync Start ({plot_data["pto_start_raw"]:.2f}s)')
        
        ax1.set_xlabel('Time (s) - Independent Timelines')
        ax1.set_ylabel('DOF: LED_On', color='blue')
        ax1_twin.set_ylabel('PTO: Trigger Voltage (V)', color='red')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

        ax2.set_title('After Alignment (PTO Timeline Shifted)')
        ax2.plot(plot_data['dof_time'], plot_data['dof_signal'], label='DOF Signal (LED_On)', color='blue', alpha=0.8)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(plot_data['pto_time_aligned'], plot_data['pto_signal'], label='PTO Signal (TriggerPaddle_V)', color='green', linestyle='--', alpha=0.8)
        ax2.axvline(plot_data['dof_start'], color='purple', linestyle='-', lw=2, label=f'Common Sync Start ({plot_data["dof_start"]:.2f}s)')
        
        ax2.set_xlabel('Time (s) - Master Timeline (from DOF)')
        ax2.set_ylabel('DOF: LED_On', color='blue')
        ax2_twin.set_ylabel('PTO: Trigger Voltage (V)', color='green')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')

        canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _run_sync_thread(self):
        self.sync_button.config(state='disabled'); self.analyze_button.config(state='disabled'); self.visualize_button.config(state='disabled')
        threading.Thread(target=self._run_synchronization, daemon=True).start()
    
    def _run_synchronization(self):
        try:
            synchronizer = DataSynchronizer(verbose=True)
            synchronizer.load_dof_data(Path(self.dof_path.get()))
            if self.gauge_path.get(): synchronizer.load_gauge_data(Path(self.gauge_path.get()))
            if self.pto_path.get(): synchronizer.load_pto_data(Path(self.pto_path.get()))
            final_df, _ = synchronizer.synchronize()
            if not final_df.empty: self.after(0, self._save_file, final_df)
            else: logging.error("Synchronization failed or resulted in an empty dataframe.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        finally:
            self.sync_button.config(state='normal'); self.analyze_button.config(state='normal')
            if self.pto_path.get(): self.visualize_button.config(state='normal')

    def _save_file(self, df):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Synchronized Data")
        if save_path:
            df.to_csv(save_path, index=False)
            logging.info(f"Successfully saved data to: {Path(save_path).name}")
            messagebox.showinfo("Success", f"Data successfully saved to:\n{save_path}")
        else:
            logging.warning("Save operation cancelled.")

if __name__ == '__main__':
    app = SyncApp()
    app.mainloop()