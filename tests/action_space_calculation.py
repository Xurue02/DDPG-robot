import sys
sys.path.append('../')  # Adjust the path accordingly

import numpy as np
import matplotlib.pyplot as plt
from kinematics.pcc_forward import trans_matrix, multiple_trans_matrix, two_section_robot
import math
'''
k = np.arange(-(5 * math.pi) / 3, (5 * math.pi) / 3, 0.1)
l = 0.06  # m
## Randomly choose two phi values within the range

np.random.seed(42)
# Generate values for phi in radians within the range from -180 to 180 degrees
phi = np.radians(np.random.uniform(low=-180, high=180))
#phi = 0

# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trunk with three sections and point the section separation
ax.plot([-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01], 'black', linewidth=3, label='Section Separation')

# Simulation for seeing task space
for k_val in k:
    # Segment 1
    T1 = trans_matrix(k_val, l, phi)
    T1_tip = np.reshape(T1[len(T1) - 1, :], (4, 4), order='F')

    # Segment 2
    T2_cc = trans_matrix(k_val, l, phi)
    T2 = multiple_trans_matrix(T2_cc, T1_tip)
    T2_tip = np.reshape(T2[len(T2) - 1, :], (4, 4), order='F')

    # Plot points for T1 and T2 linewidth=3, marker='o',
    ax.plot(T1[:, 12], T1[:, 13], T1[:, 14],  color='blue')
    ax.plot(T2[:, 12], T2[:, 13], T2[:, 14],  color='red')

# Set labels and title
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Task space of 3D Plot of Continuum Robot")
ax.legend(loc="best")

# Set plot limits
plt.xlim(-0.06, 0.06)
plt.ylim(-0.06, 0.06)

# Save the plot (uncomment the line below if you want to save)
# plt.savefig('../figures/3d_robot/tip.png')

# Show the plot
plt.show()
'''
# Number of points for phi and k
num_points = 100

# Generate phi values in radians ranging from -180 to 180 degrees
phi_radians = np.linspace(np.radians(-180), np.radians(180), num_points)

# Generate k values within the specified range
k_range_start = -(5 * math.pi) / 3
k_range_end = (5 * math.pi) / 3
k_values = np.linspace(k_range_start, k_range_end, num_points)

l = 0.06  # m

# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trunk with three sections and point the section separation
ax.plot([-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01], 'black', linewidth=3, label='Section Separation')

# Simulation for seeing task space
for i in range(num_points):
    phi_val = phi_radians[i]
    k_val = k_values[i]

    # Segment 1
    T1 = trans_matrix(k_val, l, phi_val)
    T1_tip = np.reshape(T1[len(T1) - 1, :], (4, 4), order='F')

    # Segment 2
    T2_cc = trans_matrix(k_val, l, phi_val)
    T2 = multiple_trans_matrix(T2_cc, T1_tip)
    T2_tip = np.reshape(T2[len(T2) - 1, :], (4, 4), order='F')

    # Plot points for T1 and T2 with labels
    ax.plot(T1[:, 12], T1[:, 13], T1[:, 14],  )
    ax.plot(T2[:, 12], T2[:, 13], T2[:, 14], )

# Set labels and title
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Task space of 3D Plot of Continuum Robot")

# Set plot limits
plt.xlim(-0.06, 0.06)
plt.ylim(-0.06, 0.06)

# Add legend
ax.legend(loc="best")

# Save the plot (uncomment the line below if you want to save)
# plt.savefig('../figures/3d_robot/tip.png')

# Show the plot
plt.show()
