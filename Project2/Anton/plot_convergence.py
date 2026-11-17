import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder containing your CSV files
directory = "Mads/Results" 
par_file = os.path.join(directory, "par_conv.csv")
seq_file = os.path.join(directory, "seq_conv.csv")

# Load the data (skipping the 'N_int,dx,error' header row)
par_data = np.genfromtxt(par_file, delimiter=',', skip_header=1)
seq_data = np.genfromtxt(seq_file, delimiter=',', skip_header=1)

# Extract dx (column 1) and error (column 2)
par_N = par_data[:, 0]
par_dx = par_data[:, 1]
par_error = par_data[:, 2]

seq_N = seq_data[:, 0]
seq_dx = seq_data[:, 1]
seq_error = seq_data[:, 2]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the O(h^2) reference line using the dx values
plt.loglog(seq_N, seq_dx**2, label=r'$\mathcal{O}(h^2)$ Reference', linestyle='--', color='black')

# Plot the Sequential and Parallel errors
# Using different markers and a slight transparency (alpha) so overlapping points are visible
plt.loglog(seq_N, seq_error, marker='o', label='Sequential', linestyle='-', color='blue')
plt.loglog(par_N, par_error, marker='x', label='Parallel', linestyle='', color='red', markersize=8)

# Formatting
plt.xlabel('Interior nodes (N)')
plt.ylabel('Global True Error')
plt.title('Grid Convergence Study: Sequential vs Parallel')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()

# Save the figure
output_image = os.path.join(directory, 'convergence_comparison.png')
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {output_image}")