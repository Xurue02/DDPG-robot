
import sys
sys.path.append('../')
sys.path.append('../environment')
#import tensorflow as tf
#from tensorflow.keras import layers
#from env import robot_env
import numpy as np
#from pcc_calculation import T1_hole,T2_hole
'''
env = robot_env()

num_states = env.observation_space.shape[0] * 2 # multiply by 2 because we have also goal state
print("Size of State Space ->  {}".format(num_states));
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions));
print('actionspace',env.action_space)

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound));
print("Min Value of Action ->  {}".format(lower_bound));






def cable_len(T1_hole,T2_hole):
    l1_len, l2_len, l3_len, l4_len, l5_len, l6_len = 0, 0, 0, 0, 0, 0
    T1_reshaped = np.array(T1_hole).reshape(5, 3, 4)
    for i in range(4):
        l1_len += np.linalg.norm(T1_reshaped[i+1, 0, :3] - T1_reshaped[i, 0, :3])
        l2_len += np.linalg.norm(T1_reshaped[i+1, 1, :3] - T1_reshaped[i, 1, :3])
        l3_len += np.linalg.norm(T1_reshaped[i+1, 2, :3] - T1_reshaped[i, 2, :3])

    T2_reshaped = np.array(T2_hole).reshape(10, 3, 4)
    #T2_reshaped = np.delete(T2_reshaped, 4, axis=0)
    for i in range(9):        
        l4_len += np.linalg.norm(T2_reshaped[i+1, 0, :3] - T2_reshaped[i, 0, :3])
        l5_len += np.linalg.norm(T2_reshaped[i+1, 1, :3] - T2_reshaped[i, 1, :3])
        l6_len += np.linalg.norm(T2_reshaped[i+1, 2, :3] - T2_reshaped[i, 2, :3])
    
    return l1_len,l2_len,l3_len, l4_len, l5_len, l6_len

l6_len = cable_len(T1_hole,T2_hole)
print("Total distance:", l6_len)

    
    #print('T1_reshaped',T1_reshaped)
    #print('T2_reshaped',T2_reshaped)

'''

T1_reshaped = np.array([
    [[-0.09132689, 0.34083659, 0.0, 1.0],
     [-0.2495097, -0.2495097, 0.0, 1.0],
     [0.34083659, -0.09132689, 0.0, 1.0]],

    [[-0.0892637, 0.34083659, 0.0670743, 1.0],
     [-0.24714746, -0.2495097, 0.07679641, 1.0],
     [0.34208275, -0.09132689, 0.040513, 1.0]],

    [[-0.08308195, 0.34083659, 0.13389499, 1.0],
     [-0.24006969, -0.2495097, 0.15330246, 1.0],
     [0.34581655, -0.09132689, 0.08087282, 1.0]],

    [[-0.072805, 0.34083659, 0.20020942, 1.0],
     [-0.22830315, -0.2495097, 0.22922885, 1.0],
     [0.35202384, -0.09132689, 0.12092685, 1.0]],

    [[-0.05847171, 0.34083659, 0.26576684, 1.0],
     [-0.21189231, -0.2495097, 0.30428852, 1.0],
     [0.36068117, -0.09132689, 0.16052365, 1.0]]
])

l1 = T1_reshaped[:, 0, :3]

# Calculate distances and sum them up
sum_of_distances1 = np.sum(np.linalg.norm(l1[1:] - l1[:-1], axis=1))
print("Sum of Distances1:", sum_of_distances1)
##5*3*4
l1 = T1_reshaped[:, 0, :3]
l2 = T1_reshaped[:, 1, :3]
l3 = T1_reshaped[:, 2, :3]

point1 = l1[0, :]  # First point
point2 = l1[1, :]  # Second point
point3 = l1[2, :]  # Third point
point4 = l1[3, :]  # Fourth point
point5 = l1[4, :]  # Fifth point

distance_1_to_2 = np.linalg.norm(point2 - point1)
distance_2_to_3 = np.linalg.norm(point3 - point2)
distance_3_to_4 = np.linalg.norm(point4 - point3)
distance_4_to_5 = np.linalg.norm(point5 - point4)

# Print distances

sum_of_distances = distance_1_to_2+distance_2_to_3 + distance_3_to_4 + distance_4_to_5

# Print the sum of distances
print("Sum of Distances:", sum_of_distances)