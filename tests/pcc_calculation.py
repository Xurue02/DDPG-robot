# from configuration space (k, length) to task space (x,y)

# %% import necessary libraries
import sys # to include the path of the package
sys.path.append('../')
#from continuum_robot.utils import *
import numpy as np
import matplotlib.pyplot as plt
import math
from kinematics.pcc_forward import trans_matrix,multiple_trans_matrix,two_section_robot
#from datetime import datetime

'''
## Enter two k values within the range
k1=-4.5;
k2=8.2;
phi1=1.57
phi2=3

if k1 > (5*math.pi)/3 or k1 < -(5*math.pi)/3:
    print("Please enter the First Curvature values between -5.235 and 5.235")
    k1 = 0;

elif k2 > (5*math.pi)/3 or k2 < -(5*math.pi)/3:
    print("Please enter the Second Curvature values between -5.235 and 5.235")
    k2 = 0;

else:
    print("Curvature Values for Each Segment are Appropriate")
'''
l = 0.06 # meter, same length for every segment

## Randomly choose two k values within the range
k_range = (-(5 * math.pi) / 3, (5 * math.pi) / 3)
k1, k2 = np.random.uniform(*k_range, size=2)
# Constraint for the curvature
k1 = max(-(5 * math.pi) / 3, min(k1, (5 * math.pi) / 3))
k2 = max(-(5 * math.pi) / 3, min(k2, (5 * math.pi) / 3))
print(f"Randomly chosen k1: {k1}")
print(f"Randomly chosen k2: {k2}")

## Randomly choose two phi values within the range
## Set the seed for reproducibility
np.random.seed(42)
# Generate two random values for phi in radians within the range from -180 to 180 degrees
phi1 = np.radians(np.random.uniform(low=-180, high=180))
phi2 = np.radians(np.random.uniform(low=-180, high=180))

print(f"Randomly selected phi1: {phi1:.4f} radians")
print(f"Randomly selected phi2: {phi2:.4f} radians")


# segment 1
T1 = trans_matrix(k1,l,phi1) #get transformation matrix reshaped in [1*16] in n array within length l and with size
T1_tip = np.reshape(T1[len(T1)-1,:],(4,4),order='F'); #reshape to 4*4 matrix
#print('T1 transmatrix\n',T1);
#print('T1_tip\n',T1_tip);
#print(T1[0,12],T1[0,13],T1[1,0],T1[1,12],T1[1,13])

# segment 2
T2_cc = trans_matrix(k2,l,phi2);#get reshaped transformation matrix of the section 2 
T2 = multiple_trans_matrix(T2_cc,T1_tip); # multiply T1 and T2 to get the robot transformation matrix
T2_tip = np.reshape(T2[len(T2)-1,:],(4,4),order='F');# reshape to 4*4 matrix
#print('T2 transmatrix\n',T2);
print('tip of the robot\n',T2_tip);
l=[0.06,0.06];

Tip_of_Rob = two_section_robot(k1,k2,l,phi1,phi2)
print('tip of robo should be same as tip of the robot as above\n',Tip_of_Rob)

# Plot the 3D diagram python pcc_calculation.py
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot points for T1
ax.plot(T1[:, 12], T1[:, 13], T1[:, 14], label="First Section", color='blue', linewidth=3, marker='o')
# Plot points for T2
ax.plot(T2[:, 12], T2[:, 13], T2[:, 14], label="Second Section", color='red', linewidth=3, marker='o')

#end-effector of the robot
ee = len(T2) - 1 
ax.text(T2[ee, 12], T2[ee, 13], T2[ee, 14], f'({T2[ee, 12]:.2f}, {T2[ee, 13]:.2f}, {T2[ee, 14]:.2f})', fontsize=8)

# add k and phi values on diagram
ax.text(T1[-1, 12], T1[-1, 13], T1[-1, 14], f'k1={k1:.2f},\n phi1={phi1:.4f}', fontsize=8, ha='right', va='bottom')
ax.text(T2[-1, 12], T2[-1, 13], T2[-1, 14], f'k2={k2:.2f},\n phi2={phi2:.4f}', fontsize=8, ha='right', va='bottom')


# Set labels and title
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("3D Plot of Continuum Robot Forward Kinematics")
ax.legend(loc="best")
plt.xlim(-0.06,0.06)
plt.ylim(-0.06,0.06)
plt.savefig('../figures/3d_robot/tip.png')
plt.show()

# %%
