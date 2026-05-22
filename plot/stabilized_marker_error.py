import matplotlib.pyplot as plt
import numpy as np

# Set plot style for scientific publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# --- Data Parsing ---
# Non-Stabilised Reference Data
non_stabilised_r_pos = np.array([
    [0.0040, 0.0032, 0.0138], [0.0041, 0.0028, 0.0130],
    [0.0035, 0.0025, 0.0109], [0.0034, 0.0021, 0.0119],
    [0.0033, 0.0017, 0.0115], [0.0031, 0.0014, 0.0079],
    [0.0035, 0.0012, 0.0076], [0.0037, 0.0010, 0.0066]
])

# Filtered Reference Data
filtered_r_pos = np.array([
    [0.0044, 0.0020, 0.0120], [0.0029, 0.0020, 0.0097],
    [0.0027, 0.0017, 0.0094], [0.0026, 0.0013, 0.0077],
    [0.0026, 0.0011, 0.0072], [0.0029, 0.0010, 0.0068],
    [0.0029, 0.0008, 0.0056], [0.0030, 0.0006, 0.0053]
])

# --- Calculate Performance Index (Percentage Reduction) ---
# Formula: ((Old - New) / Old) * 100
# For means
mean_non_stabilised = np.mean(non_stabilised_r_pos, axis=0)
mean_filtered = np.mean(filtered_r_pos, axis=0)
mean_reduction_percent = (1 - mean_filtered / mean_non_stabilised) * 100

# For individual buoys
individual_reduction_percent = (1 - filtered_r_pos / non_stabilised_r_pos) * 100

# --- Plotting ---
buoy_labels = [f'Buoy {i+1}' for i in range(8)]
x = np.arange(len(buoy_labels))
width = 0.35
color1 = 'cornflowerblue'
color2 = 'darkorange'

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# --- Subplot for each axis ---
axis_labels = ['X', 'Y', 'Z']
panel_labels = ['A', 'B', 'C']

for i in range(3):
    ax = axes[i]
    # Bar plots
    ax.bar(x - width/2, non_stabilised_r_pos[:, i], width, label='Non-Stabilised Ref.', color=color1)
    ax.bar(x + width/2, filtered_r_pos[:, i], width, label='Filtered Ref.', color=color2)

    # Add text box with performance index
    bbox_props = dict(boxstyle="round,pad=0.3", fc="ivory", ec="black", lw=1, alpha=0.9)
    ax.text(0.97, 0.95, f'Mean Error Reduction: {mean_reduction_percent[i]:.2f}%',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=bbox_props)

    # Formatting
    ax.set_ylabel(f'Error Variance ($R_{{{axis_labels[i].lower()}}}$)')
    ax.set_title(f'({panel_labels[i]}) {axis_labels[i]}-Axis Position Error Comparison', loc='left', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False)
    if i == 0:
        ax.legend(loc='upper left')

# Shared X-axis settings
axes[-1].tick_params(axis='x', which='both', bottom=True)
plt.xticks(x, buoy_labels, rotation=0)
plt.xlabel('Buoy Identifier')
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the figure
plt.savefig("buoy_error_comparison_indexed.png", dpi=300, bbox_inches='tight')

plt.show()