# DDPG-robot
## Project Overview
This project is designed for Cable Driven Continuum Robot with Reinforcement Learning Based Control.
## Setup and Installation

Python 3.x (3.9.18) required. Install dependencies via:

1. Create the environment from the `Env.yml` file:
  `conda env create -f environment.yml`

2. Activate the new environment: `conda activate continuum` 

Alternative way to creat environment: `conda active "your-own-env-name"` and then `pip install -r requirements.txt`.

## DDPG part code
 - `environment`file: This file contains the environment that the agent(robot) has interactions with, `env.py` includes the code for action space, observation space and rewards function.
 - `kinematics` file: This file contains the content of how coordinates of robot's end-effector is obtained by taking robot parameters as input into PCC forward kinematic.
 - `plotsfunc`file: This file describes the code of plotting reward saved in pickle form.
 - `tensorflow`file: This file contains the ddpg training stage code , weights of four networks after training and rewards per training episode.
 - `tests`file: This file contains the code of tesing pcc forward kinematics and ddpg trained model.

## Matlab Code.zip
 - This zip file contains the Matlab code of the project, which includes the PCC forward kinematic calculation, forward kinematics calculation and simulation, IMU based shape reconstrunction and vision based validation.

 ## Motor Code.zip
 - This zip file contains Python and Arduino codes associated with controlling the motor. Including the file motorPID.ino which should be uploaded to the microcontroller, targets_input.py is for manual target input, gesture_control.py and handdetector.py are for the first supplementary control method where controller_serial.py is for the second supplementary control method.
