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
        self.delta_k = 0.001     # necessary for the numerical differentiation
        self.k_dot_max = 1.000   # max derivative of curvature
        self.k_max = 16.00       # max curvature for the robot
        self.k_min = -4.00       # min curvature for the robot
        
        l1 = 0.06000;               # first segment of the robot in meters
        l2 = 0.06000;               # second segment of the robot in meters
        self.stop = 0               # variable to make robot not move after exeeding max, min general k value
        self.l = [l1, l2]           # stores the length of each segment of the robot
        self.dt =  5e-2             # sample sizes
        #self.J = np.zeros((2,3)    # initializes the Jacobian matrix  
        self.error = 0              # initializes the error
        self.previous_error = 0     # initializes the previous error
        self.start_k = [0,0]        # initializes the start curvatures for the two segments
        self.start_phi = [0,0]      # initializes the start phi for the two segments
        self.time = 0               # to count the time of the simulation
        self.overshoot0 = 0
        self.overshoot1 = 0
        self.position_dic = {'Section1': {'x':[],'y':[],'z':[]}, 'Section2': {'x':[],'y':[],'z':[]}}

        # Define the observation and action space from OpenAI Gym
        high = np.array([0.2, 0.3, 0.16, 0.3], dtype=np.float32) # [0.16, 0.3, 0.16, 0.3]
        low  = np.array([-0.3, -0.15, -0.27, -0.11], dtype=np.float32) # [-0.27, -0.11, -0.27, -0.11]
        self.action_space = spaces.Box(low=-1*self.k_dot_max, high=self.k_dot_max,shape=(3,), dtype=np.float32)
        ########
        
        # TODO: Add better environment observation space (more circle or algorithm that make automatically)
        self.observation_space = (space)

    def reward_calculation(self,u): 
        '''
          The reward is designed to be the negative square of the Euclidean distance between the current 
          position of the robot and its goal position.

          Reward is -(e^2)
        '''
        
        x,y,goal_x,goal_y = self.state # Get the current state as x,y,goal_x,goal_y
        dt =  self.dt # Time step
        
        u = np.clip(u, -self.k_dot_max, self.k_dot_max) # Clip the input to the range of the -1,1
        
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
         
        
        # This if and else statement is to avoid the robot to move if the ks are at the limits
        if self.stop == 0:
            self.J = jacobian_matrix(self.delta_k, self.k1, self.k2, self.k3, self.l)
            x_vel = self.J @ u
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 1:
            self.J = jacobian_matrix(self.delta_k, self.k1, self.k2, self.k3, self.l)
            x_vel = self.J @ np.append([0],u[1:3])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 2:
            self.J = jacobian_matrix(self.delta_k, self.k1, self.k2, self.k3, self.l)
            x_vel = self.J @ np.append(np.append(u[0],[0]),u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 3:
            self.J = jacobian_matrix(self.delta_k, self.k1, self.k2, self.k3, self.l)
            x_vel = self.J @ np.append(u[0:2],[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
            
        elif self.stop == 4:
            self.J = jacobian_matrix(self.delta_k, self.k1, self.k2, self.k3, self.l)
            x_vel = self.J @ np.append([0,0],u[2])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 5:
            self.J = jacobian_matrix(self.delta_k, self.k1, self.k2, self.k3, self.l)
            x_vel = self.J @  np.append(np.append([0],u[1]),[0])
            state_update = x_vel * dt
            new_x = x + state_update[0]
            new_y = y + state_update[1]
        
        elif self.stop == 6:
            self.J = jacobian_matrix(self.delta_k, self.k1, self.k2, self.k3, self.l)
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

        # TODO -> Solve the situation when ks are zero in Homogenous matrix
        # Maybe when it is Zero try except and Raise an error
        self.k1 = np.clip(self.k1, self.k_min, self.k_max)
        self.k2 = np.clip(self.k2, self.k_min, self.k_max)

        # To check which curvature value are at the limits
        self.stop = 0
        k1 = self.k1 <= self.k_min or self.k1 >= self.k_max
        k2 = self.k2 <= self.k_min or self.k2 >= self.k_max
        
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
        goal_x,goal_y,goal_z = np.array([Tip_target[0,3],Tip_target[1,3],Tip_target[2,3]]) # Extract the x and y coordinates of the target
       
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
        self.position_dic['Section2']['z'].append(T1[:,14])

    def render_init(self):
        # This function is used to plot the robot in the environment (both in start and end state)
        self.fig = plt.figure()
        self.fig.set_dpi(75);
        self.ax = plt.axes();
        
    def render_init(self):
        self.ax = plt.axes();

    def render_update(self,i):
        self.ax.cla()
        # Plot the trunk with three sections and point the section seperation
        self.ax.plot([-0.025, 0.025],[0,0],'black',linewidth=5)
        self.ax.plot(self.position_dic['Section1']['x'][i],self.position_dic['Section1']['y'][i],'b',linewidth=3)
        #plt.scatter(T1_cc[-1,12],T1_cc[-1,13],linewidths=5,color = 'black')
        self.ax.plot(self.position_dic['Section2']['x'][i],self.position_dic['Section2']['y'][i],'r',linewidth=3)
        #plt.scatter(T2_cc[-1,12],T2_cc[-1,13],linewidths=5,color = 'black')
        self.ax.plot(self.position_dic['Section3']['x'][i],self.position_dic['Section3']['y'][i],'g',linewidth=3)
        self.ax.scatter(self.position_dic['Section3']['x'][i][-1],self.position_dic['Section3']['y'][i][-1],linewidths=5,color = 'black')

        # Plot the target point and trajectory of the robot
        self.ax.scatter(self.state[2],self.state[3],100, marker= "x",linewidths=2, color = 'red')
        self.ax.set_title(f"The time elapsed in the simulation is {round(self.time,2)} seconds.")
        self.ax.set_xlabel("X - Position [m]")
        self.ax.set_ylabel("Y - Position [m]")
        self.ax.set_xlim([-0.4, 0.4])
        self.ax.set_ylim([-0.4, 0.4])

        # Plot the 3D diagram python pcc_calculation.py
        self.fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        # Plot points for T1
        self.ax.plot(self.position_dic['Section1']['x'][i],self.position_dic['Section1']['y'][i],self.position_dic['Section1']['z'][i],'b',linewidth=3)
        #self.ax.plot(T1[:, 12], T1[:, 13], T1[:, 14], label="First Section", color='blue', linewidth=3, marker='o')
        # Plot points for T2
        #self.ax.plot(T2[:, 12], T2[:, 13], T2[:, 14], label="Second Section", color='red', linewidth=3, marker='o')
        self.ax.plot(self.position_dic['Section2']['x'][i],self.position_dic['Section2']['y'][i],self.position_dic['Section1']['z'][i],'r',linewidth=3)
        
        # Plot the target point of the robot
        self.ax.scatter(self.state[3],self.state[4],self.state[5],100, marker= "x",linewidths=2, color = 'red')
        # Set labels and title
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title("3D Plot of Continuum Robot Forward Kinematics")
        self.ax.legend(loc="best")
        self.plt.xlim(-0.06,0.06)
        self.plt.ylim(-0.06,0.06)
        self.plt.savefig('../figures/3d_robot/tip_of_rob.png')
        self.plt.show()

    def visualization(self,x_pos,y_pos,z_pos):
        #Start state
        #start curvatures are set above: self.start_k = [0,0]
        #start phi are set above: self.start_phi = [0,0]; and length: self.l = [l1, l2]
        # segment 1 
        
        T1 = trans_matrix(self.start_kappa[0],self.l[0],self.start_phi[0]) #get transformation matrix reshaped in [1*16] in n array within length l and with size
        T1_tip = np.reshape(T1[len(T1)-1,:],(4,4),order='F'); #reshape to 4*4 matrix

        # segment 2
        T2_cc = trans_matrix(self.start_kappa[1],self.l[1],self.start_phi[1]);#get reshaped transformation matrix of the section 2 
        T2 = multiple_trans_matrix(T2_cc,T1_tip); # multiply T1 and T2 to get the robot transformation matrix
        T2_tip = np.reshape(T2[len(T2)-1,:],(4,4),order='F');# reshape to 4*4 matrix

        # Plot the 3D diagram python pcc_calculation.py
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot points for T1
        ax.plot(T1[:, 12], T1[:, 13], T1[:, 14], label="First Section", color='blue', linewidth=3, marker='o')
        # Plot points for T2
        ax.plot(T2[:, 12], T2[:, 13], T2[:, 14], label="Second Section", color='red', linewidth=3, marker='o')

        #End state
        # segment 1
        T1 = trans_matrix(self.k1,self.l[0],self.phi1) #get transformation matrix reshaped in [1*16] in n array within length l and with size
        T1_tip = np.reshape(T1[len(T1)-1,:],(4,4),order='F'); #reshape to 4*4 matrix

        # segment 2
        T2_cc = trans_matrix(self.k2,self.l[0],self.phi2);#get reshaped transformation matrix of the section 2 
        T2 = multiple_trans_matrix(T2_cc,T1_tip); # multiply T1 and T2 to get the robot transformation matrix
        T2_tip = np.reshape(T2[len(T2)-1,:],(4,4),order='F');# reshape to 4*4 matrix
        
        # Plot the 3D diagram python pcc_calculation.py
        # Plot points for T1
        ax.plot(T1[:, 12], T1[:, 13], T1[:, 14], label="First Section", color='blue', linewidth=3, marker='o')
        # Plot points for T2
        ax.plot(T2[:, 12], T2[:, 13], T2[:, 14], label="Second Section", color='red', linewidth=3, marker='o')
        # Plot the target point of the robot
        ax.scatter(self.state[3],self.state[4],self.state[5],100, marker= "x",linewidths=2, color = 'red')
        # Set labels and title
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title("3D Plot of Continuum Robot")
        self.ax.legend(loc="best")
        self.plt.xlim(-0.06,0.06)
        self.plt.ylim(-0.06,0.06)
        self.plt.savefig('../figures/3d_robot/visualization.png')
        self.plt.show()
