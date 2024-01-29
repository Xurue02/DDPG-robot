import sys
sys.path.append('../')
sys.path.append('../environment')
sys.path.append('../tests')

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from env import robot_env

env = robot_env()

num_states = env.observation_space.shape[0] * 2 # multiply by 2 because we have also goal state
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
start_time = time.time()


class OUActionNoise:
    '''
    It creates a correlated noise with mean-reverting behavious based on provious value; 
    injecting controlled randomness into the actions of an agen can promote exploration and improve learning performance
    '''

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        '''
        initializes the parameters of the OU process, with an optional initial vallue
        '''
        self.theta = theta           # the rate at which the process reverts to the mean
        self.mean = mean             # the mean value of the moise process
        self.std_dev = std_deviation # the standard deviation of the noise process
        self.dt = dt                 # the time step used in simulation
        self.x_initial = x_initial   # the initial value of the noise process, =0 if not provided
        self.reset()
    
    
    def __call__(self):
        '''
        Implements the Ornstein-Uhlenbeck process formula,update the noise value based on the
        prievious value, the mean and the SD
        '''
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev        
        self.x_prev = x
        return x
    
    def reset(self):
        '''
        Resets the noise process to its initial state. If an initial value is provided,
        it uses that value; otherwise, it resets to a zero vector.
        '''
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class Buffer:
    """
    Replay buffer for Deep Deterministic Policy Gradient (DDPG) algorithm.
    """
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Maximum number of experiences to store
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on in each learning step.
        self.batch_size = batch_size
        # Tracks the number of times the record() has been called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded, replacing old records
        index = self.buffer_counter % self.buffer_capacity

        # Unpack and store the observation tuple
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        # Increment the buffer counter
        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        # ===================================================================== #
		#                               Actor Model                             #
        # ===================================================================== #
        #                               Critic Model                            #
        # ===================================================================== #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        """
        Learn from a batch of sampled experiences.
        """
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        # Call the update method to perform the learning step
        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

#       # ========================================================================= #
	    #                              Model Definitions                            #
        # ========================================================================= #

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003) # minval=-0.003, maxval=0.003
    # Define the input layer with the shape of the state space
    inputs = layers.Input(shape=(num_states,))

    # hidden layer with 256 units and ReLU activation
    out = layers.Dense(256, activation="relu")(inputs) # 256  
    # Another hidden layer with 256 units and ReLU activation
    out = layers.Dense(256, activation="relu")(out) # 256
      
    # Output layer with 3 actions: num_actions units and tanh activation, initialized with last_init
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 1.0 for Continuum Robot (k dot).
    # Scale the outputs by the upper_bound to match the action space
    outputs = outputs * upper_bound # * env.dt

    # Create the Keras model with the specified inputs and outputs
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    # Neural network for processing state input
    state_out = layers.Dense(16, activation="relu")(state_input) 
    state_out = layers.Dense(32, activation="relu")(state_out) 
    

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    # Neural network for processing action input
    action_out = layers.Dense(32, activation="relu")(action_input)
    
    # Concatenate processed state and action representations
    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    # Neural network architecture for the concatenated representation
    out = layers.Dense(256, activation="relu")(concat) # 256
    out = layers.Dense(256, activation="relu")(out) # 256

    # Output layer, predicting the Q-value for the given state-action pair
    outputs = layers.Dense(1)(out) 

    # Create the Keras model with the specified inputs and outputs
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, noise_object,add_noise=True):
    # Get the output from the actor model for the given state
    sampled_actions = tf.squeeze(actor_model(state))

    # Generate noise using the noise_object
    noise = noise_object() # change to 0 to delete noise

    # # Adding noise to the sampled actions if specified
    if add_noise:
        sampled_actions = sampled_actions.numpy() + noise
    
    # Clip the actions to ensure they are within the specified bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

total_episodes = 250
# Discount factor for future rewards
gamma = 0.99            # discount factor
# Used to update target networks
tau = 5e-3              # for soft update of target parameters

buffer = Buffer(int(5e5), 128) # Buffer(50000, 128)

# %% Train or Evaluate
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
counter = 0
avg_reward = 0

TRAIN = False

if TRAIN:
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions))
    
    for ep in range(total_episodes):
        
        # prev_state = env.reset_known() # starting position is always same
        prev_state = env.reset() # starting postion is random (within task space)
        # high = np.array([0.18, 0.3], dtype=np.float32)
        # low = np.array([-0.25, -0.1], dtype=np.float32)
        # env.q_goal = np.random.uniform(low=low, high=high)
        # x y xg yg    x y z xg yg zg
        # 0 1 2  3     0 1 2 3  4  5
        if ep % 100 == 0:
            print('Episode Number',ep)
            print("Initial Position is",prev_state[0:3])
            print("===============================================================")
            print("Target Position is",prev_state[3:6])
            print("===============================================================")
            print("Initial curvatures are ",[env.k1,env.k2])
            print("===============================================================")
            print("Goal curvatures are ",[env.target_k1,env.target_k2])
            print("===============================================================")
        
        # time.sleep(2) # uncomment when training in local computer
        episodic_reward = 0
    
        # while True:
        for i in range(1000):
            # env.render()
    
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = policy(tf_prev_state, ou_noise) #get action
    
            # Recieve state and reward from environment.
            state, reward, done, info = env.reward_calculation(action[0]) # reward is -e^2
            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
    
            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
    
            # End this episode when `done` is True
            if done:
                counter += 1
                break
    
            prev_state = state

            if i % 200 == 0: # print on terminal every 200 actions.
                print("Episode Number {0} and {1}th action".format(ep,i))
                print("Goal Position",prev_state[3:6])
                print("Previous Error: {0}, Error: {1}, Current State: {2}".format(env.previous_error, env.error, prev_state[0:3])) 
                print("Action: {0},  Reward is {1}".format(action, reward))
            
    
        ep_reward_list.append(episodic_reward)
    
        # Mean of 250 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        if ep % 100 == 0:
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            time.sleep(0.5)
        avg_reward_list.append(avg_reward)
    
    print(f'{counter} times robot reached the target point in total {total_episodes} episodes')
    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(avg_reward_list)+1), avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")

    with open('avg_reward_list.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(avg_reward_list, f, pickle.HIGHEST_PROTOCOL)
    
    # Episodes versus Rewards
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(ep_reward_list)+1), ep_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()
    plt.savefig('../figures/training_rewards/rewards.png')

    with open('ep_reward_list.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(ep_reward_list, f, pickle.HIGHEST_PROTOCOL)
    
    # Save Weights
    actor_model.save_weights("continuum_actor.h5")
    critic_model.save_weights("continuum_critic.h5")
    target_actor.save_weights("continuum_target_actor.h5")
    target_critic.save_weights("continuum_target_critic.h5")
    end_time = time.time() - start_time
    print('Total Overshoot 0: ', env.overshoot0)
    print('Total Overshoot 1: ', env.overshoot1)
    print('Total Elapsed Time is:',int(end_time)/60)

else:
    actor_model.load_weights("../tensorflow/fixed_goal/model/continuum_actor.h5")
    critic_model.load_weights("../tensorflow/fixed_goal/model/continuum_critic.h5")
    target_actor.load_weights("../tensorflow/fixed_goal/model/continuum_target_actor.h5")
    target_critic.load_weights("../tensorflow/fixed_goal/model/continuum_target_critic.h5")