import sys # to include the path of the package
sys.path.append('../') # the kinematics functions are here 

import gym                      # openai gym library
import numpy as np              # numpy for matrix operations
import math                     # math for basic calculations
from gym import spaces          # "spaces" for the observation and action space
import matplotlib.pyplot as plt # quick "plot" library
from matplotlib.animation import FuncAnimation # make animation
from kinematics.pcc_forward import trans_matrix,multiple_trans_matrix,two_section_robot
#from AmorphousSpace import AmorphousSpace

class robot_env(gym.Env):
    def __init__(self):

    def reward_calculation(self,u):# reward is -(e^2)

    def reset(self):
