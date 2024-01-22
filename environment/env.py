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
        self.delta_kappa = 0.001     # necessary for the numerical differentiation
        self.kappa_dot_max = 1.000  # max derivative of curvature
        self.kappa_max = 16.00      # max curvature for the robot
        self.kappa_min = -4.00      # min curvature for the robot
        
        l1 = 0.06000;                # first segment of the robot in meters
        l2 = 0.06000;                # second segment of the robot in meters
        self.stop = 0               # variable to make robot not move after exeeding max, min general kappa value
        self.l = [l1, l2]           # stores the length of each segment of the robot
        self.dt =  5e-2             # sample sizes
        #self.J = np.zeros((2,3))    # initializes the Jacobian matrix  
        self.error = 0              # initializes the error
        self.previous_error = 0     # initializes the previous error
        self.start_kappa = [0,0]  # initializes the start kappas for the three segments
        self.time = 0               # to count the time of the simulation
        self.overshoot0 = 0
        self.overshoot1 = 0
        self.position_dic = {'Section1': {'x':[],'y':[]}, 'Section2': {'x':[],'y':[]}}

        # Define the observation and action space from OpenAI Gym
        high = np.array([0.2, 0.3, 0.16, 0.3], dtype=np.float32) # [0.16, 0.3, 0.16, 0.3]
        low  = np.array([-0.3, -0.15, -0.27, -0.11], dtype=np.float32) # [-0.27, -0.11, -0.27, -0.11]
        self.action_space = spaces.Box(low=-1*self.kappa_dot_max, high=self.kappa_dot_max,shape=(3,), dtype=np.float32)
        ########
        
        # TODO: Add better environment observation space (more circle or algorithm that make automatically)
        self.observation_space = (space)

    def reward_calculation(self,u): 
        '''
          The reward is designed to be the negative square of the Euclidean distance between the current position of the robot and its goal position
        '''
        # reward is -(e^2)
        x,y,goal_x,goal_y = self.state # Get the current state as x,y,goal_x,goal_y
        dt =  self.dt # Time step
        
        u = np.clip(u, -self.kappa_dot_max, self.kappa_dot_max) # Clip the input to the range of the -1,1
        
        self.error = ((goal_x-x)**2)+((goal_y-y)**2) # Calculate the error squared
        self.costs = self.error # Set the cost (reward) to the error squared
        
        # Just to show if the robot is moving along the goal or not
        if self.error < self.previous_error:
            pass
                    
        self.previous_error = self.error 
        
        # if the error is less than 0.01, the robot is close to the goal and returns done
        if math.sqrt(self.costs) <= 0.01:
            done = True
        else :
            done = False
         
        
        # This if and else statement is to avoid the robot to move if the kappas are at the limits
        if self.stop == 0:
            self.J = jacobian_matrix(self.delta_kappa, self.k1, self.k2, self.kappa3, self.l)
            x_vel = self.J @ u
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 1:
            self.J = jacobian_matrix(self.delta_kappa, self.k1, self.k2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0],u[1:3])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 2:
            self.J = jacobian_matrix(self.delta_kappa, self.k1, self.k2, self.kappa3, self.l)
            x_vel = self.J @ np.append(np.append(u[0],[0]),u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 3:
            self.J = jacobian_matrix(self.delta_kappa, self.k1, self.k2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0:2],[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 4:
            self.J = jacobian_matrix(self.delta_kappa, self.k1, self.k2, self.kappa3, self.l)
            x_vel = self.J @ np.append([0,0],u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 5:
            self.J = jacobian_matrix(self.delta_kappa, self.k1, self.k2, self.kappa3, self.l)
            x_vel = self.J @  np.append(np.append([0],u[1]),[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 6:
            self.J = jacobian_matrix(self.delta_kappa, self.k1, self.k2, self.kappa3, self.l)
            x_vel = self.J @ np.append(u[0],[0,0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 7:
            pass
            # # UNCOMMENT HERE!!!!!!!
            # print("Robot is not moving")
            # time.sleep(1)
        
        # Update the curvatures
        self.k1 += u[0] * dt 
        self.k2 += u[1] * dt      

        # TODO -> Solve the situation when kappas are zero in Homogenous matrix
        # Maybe when it is Zero try except and Raise an error
        self.k1 = np.clip(self.k1, self.kappa_min, self.kappa_max)
        self.k2 = np.clip(self.k2, self.kappa_min, self.kappa_max)

        # To check which curvature value are at the limits
        self.stop = 0
        k1 = self.k1 <= self.kappa_min or self.k1 >= self.kappa_max
        k2 = self.k2 <= self.kappa_min or self.k2 >= self.kappa_max
        
        if k1:
            self.stop = 1
            
        elif k2:
            self.stop = 2
            
        elif k3:
            self.stop = 3
        
        if k1 and k2:
            self.stop = 4
        
        elif k1 and k3:
            self.stop = 5
        
        elif k2 and k3:
            self.stop = 6
            
        if k1 and k2 and k3:
            self.stop = 7
        
        if self.observation_space.contains([new_x, new_y]):
            pass
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot0 += 1
            # print(new_x, new_y)
            new_x, new_y = self.observation_space.clip([new_x,new_y])
            # print(new_x, new_y)

        if self.observation_space.contains([goal_x, goal_y]):
            new_goal_x, new_goal_y = goal_x, goal_y
        else:
            # Clip the states to avoid the robot to go out of the workspace
            self.overshoot1 += 1
            # print(goal_x,goal_y)
            new_goal_x, new_goal_y = self.observation_space.clip([goal_x,goal_y])
            # print(new_goal_x, new_goal_y)

        # States of the robot in numpy array
        self.state = np.array([new_x,new_y,new_goal_x,new_goal_y])
        
        return self._get_obs(), -1*self.costs, done, {} # Return the observation, the reward (-costs) and the done flag

    def reset(self): 

        # Random state of the robot 
        ## Randomly choose two phi values within the range
        ## Set the seed for reproducibility
        np.random.seed(42)
        # Generate two random values for phi in radians within the range from -180 to 180 degrees
        self.phi1 = np.radians(np.random.uniform(low=-180, high=180))
        self.phi2 = np.radians(np.random.uniform(low=-180, high=180))

        # Random curvatures 
        self.k1 = np.random.uniform(low=-10, high=16)
        self.k2 = np.random.uniform(low=-10, high=16)
        # pcc calculation
        Tip_of_Rob = two_section_robot(self.k1,self.k2,self.l,self.phi1,self.phi2) 
        x,y,z = np.array([Tip_of_Rob[0,3],Tip_of_Rob[1,3],Tip_of_Rob[2,3]]) # Extract the x,y and z coordinates of the tip

        # Random target point
        # (Random curvatures are given so that forward kinematics equation will generate random target position)
        self.target_k1 = np.random.uniform(low=-10, high=16) # 6.2 # np.random.uniform(low=-4, high=16)
        self.target_k2 = np.random.uniform(low=-10, high=16) # 6.2 # np.random.uniform(low=-4, high=16)
        # pcc calculation
        Tip_target = two_section_robot(self.target_k1,self.target_k2,self.l,self.phi1,self.phi2) # Generate the target point for the robot
        goal_x,goal_y,goal_z = np.array([T3_target[0,3],T3_target[1,3],T3_target[2,3]]) # Extract the x and y coordinates of the target
       
        self.state = x,y,z,goal_x,goal_y,goal_z # Update the state of the robot
       
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        x,y,z,goal_x,goal_y,goal_z = self.state
        return np.array([x,y,z,goal_x,goal_y,goal_z],dtype=np.float32)
    
    def render_calculate(self):
        # current state

        # segment 1
        T1 = trans_matrix(self.k1,self.l[0],self.phi1) #get transformation matrix reshaped in [1*16] in n array within length l and with size
        T1_tip = np.reshape(T1[len(T1)-1,:],(4,4),order='F'); #reshape to 4*4 matrix

        # segment 2
        T2_cc = trans_matrix(self.k2,self.l[1],self.phi2);#get reshaped transformation matrix of the section 2 
        T2 = multiple_trans_matrix(T2_cc,T1_tip); # multiply T1 and T2 to get the robot transformation matrix
        T2_tip = np.reshape(T2[len(T2)-1,:],(4,4),order='F');# reshape to 4*4 matrix

        self.position_dic['Section1']['x'].append(T1[:,12])
        self.position_dic['Section1']['y'].append(T1[:,13])
        self.position_dic['Section1']['z'].append(T1[:,14])
        self.position_dic['Section2']['x'].append(T2[:,12])
        self.position_dic['Section2']['y'].append(T2[:,13])
        self.position_dic['Section1']['z'].append(T1[:,14])