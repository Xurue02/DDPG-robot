# %%
import sys
sys.path.append('../')
sys.path.append('../environment')
sys.path.append('../tensorflow')
sys.path.append('../tests')

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
from env import robot_env
from ddpg import OUActionNoise,policy
plt.style.use('../continuum_robot/plot.mplstyle')
from plotsfunc import *
from matplotlib import animation
# %matplotlib notebook
from IPython import display

storage = {store_name: {} for store_name in ['error', 'pos', 'k','reward']}
storage['error']['error_store'] = []
storage['error']['x'] = []
storage['error']['y'] = []
storage['error']['z'] = []

storage['pos']['x'] = []
storage['pos']['y'] = []
storage['pos']['z'] = []

storage['cable length']['l1'] = []
storage['cable length']['l2'] = []
storage['cable length']['l3'] = []
storage['cable length']['l4'] = []
storage['cable length']['l5'] = []
storage['cable length']['l6'] = []

storage['reward']['value'] = []
storage['reward']['rewards'] = []

episode_number = 250
counter = 0
for _ in range(episode_number):
    env = robot_env() 

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(3), std_deviation=float(std_dev) * np.ones(3))

    state = env.reset() # generate random starting point for the robot and random target point.
    env.time = 0
    initial_state = state[0:3] # state = x,y,z,goal_x,goal_y,goal_z

    env.render_init() # uncomment for animation

    N = 1000
    step = 0
    for step in range(N): # or while True:
        start = time.time()
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = policy(tf_prev_state, ou_noise, add_noise = False) 
        state, reward, done, info = env.reward_calculation(action[0]) 
        
        storage['pos']['x'].append(state[0])
        storage['pos']['y'].append(state[1])
        storage['pos']['z'].append(state[2])


        print("{}th action".format(step))
        print("Goal k:{0}, phi:{1}, l:{2}".format(env.k1,))
        print("Goal Position",state[3:6])
        print("Action: {0},  cable_lenghts {1}".format(action, env.cab_lens))
        print("Reward is ", reward)
        
        stop = time.time()
        env.time += (stop - start)
        storage['cable length']['l1'].append(env.cab_lens[0])
        storage['cable length']['l2'].append(env.cab_lens[1])
        storage['cable length']['l3'].append(env.cab_lens[2])
        storage['cable length']['l4'].append(env.cab_lens[3])
        storage['cable length']['l5'].append(env.cab_lens[4])
        storage['cable length']['l6'].append(env.cab_lens[5])
        storage['reward']['value'].append(reward)

        if done:
            counter += 1
                print(f"reached the {counter} times")
            break
    storage['reward']['rewards'].append(step)
                           
time.sleep(1)
rewards = np.mean(storage['reward']['rewards'])
print(f'Average  reward is {rewards}')

print(f'{counter} times robot reached the target point in total {total_episodes} episodes')



    
