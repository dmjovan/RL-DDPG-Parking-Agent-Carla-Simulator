# ---------------------------------------------------------------------------------------------------
# IMPORTING ALL NECESSARY LIBRARIES
# ---------------------------------------------------------------------------------------------------

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

# ---------------------------------------------------------------------------------------------------
# IMPORTING CARLA SIMULATOR
# ---------------------------------------------------------------------------------------------------
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# ---------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES FOR CARLA AND ENVIRONMENT
# ---------------------------------------------------------------------------------------------------
# host and port for Carla server side operating
_HOST_ = '127.0.0.1'
_PORT_ = 2000

_SLEEP_TIME_ = 0.5 # s

# path to this folder
FOLDER_PATH = os.getcwd()

# path to simplified map of parking
MAP_CSV_PATH = FOLDER_PATH + '/env_scripts_and_data/parking_map_for_spawn_on_'

# global variable for (en/dis)abling spawning of vehicles on the sides of goal parking spot
VEHICLES_ON_SIDE_AVAILABLE = False

# number of varibles in one state vector
ACTIONS_SIZE = 2

# number of varibles in one state vector
STATE_SIZE = 15

# ---------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES FOR TRAINING 
# ---------------------------------------------------------------------------------------------------
# switch for getting old, trained models, or creating new
LOAD_MODEL_WEIGHTS_ENABLED = False

# total number of episodes
TOTAL_EPISODES = 1000

# duration of one episode
SECONDS_PER_EPISODE = 50

# duration of one episode
NON_MOVING_SECONDS = 20

# number of episodes to get average estimate
AVERAGE_EPISODES_COUNT = 30

# size of replay buffer/memory
REPLAY_BUFFER_CAPACITY = 100000

# batch size for replay buffer
BATCH_SIZE = 32

# learning rates for actor-critic models
CRITIC_LR = 0.001
ACTOR_LR = 0.0001

# discount factor for future rewards
GAMMA = 0.99

# rate used for slowly updating target networks
TAU = 0.005 #0.005

# scheduler for epsilon
EXPLORE = 1000000

# epsilon setting
epsilon = 1

# ---------------------------------------------------------------------------------------------------
# CARLA ENVIRONMENT CLASS
# ---------------------------------------------------------------------------------------------------
# class handles with real objects in Carla siumlator, takes actions on them and calculates important values
class CarlaEnvironment:

    # initially current state is set to None
    current_state = None 

    # constructor of the class
    def __init__(self, spawn_waypoint):

        self.client = carla.Client(_HOST_, _PORT_)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town05') 
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        self.model_3 = self.blueprint_library.filter('model3')[0]
        # https://www.automobiledimension.com/photos/tesla-model-3-2018.jpg <---------- image with Tesla Model 3 dimensions
        self.model_3_heigth = 1.443 # m
        self.model_3_length = 4.694 # m
        self.model_3_width = 2.089 # m

        # offset so that there wont be any collision on start
        self.spawning_z_offset = 0.5

        # getting simplified parking map
        self.start_transform, self.spectator_transform, self.parking_map = self.get_parking_map(spawn_waypoint)

        # all 8 radar readings - set to max value od 20 meters
        self.radar_readings = {
                                'radar_0'  : 20.0,
                                'radar_45' : 20.0,
                                'radar_90' : 20.0,
                                'radar_135': 20.0,
                                'radar_180': 20.0,
                                'radar_225': 20.0,
                                'radar_270': 20.0, 
                                'radar_315': 20.0 
                               }

        # setting transform of spectator for observing process
        self.world.get_spectator().set_transform(self.spectator_transform)

    # function for getting real spots on parking lot from previously catched data while observing environment / Town05
    def get_parking_map(self, spawn_waypoint):

        full_path = MAP_CSV_PATH + spawn_waypoint + '.csv'
        
        # reading previously formed parking.csv file
        df = pd.read_csv(full_path, index_col = ['position'])
        df = df.apply(pd.to_numeric, errors='coerce')

        # starting transform
        start_x, start_y, start_rotation = df.loc['start', 'x'], df.loc['start', 'y'], df.loc['start','yaw']

        # creating transform for starting
        start_transform = carla.Transform(carla.Location(x=start_x, y=start_y, z=self.spawning_z_offset),
                                          carla.Rotation(yaw=start_rotation))

        # getting transform for spectator
        spec_x, spec_y, spec_z, spec_yaw, spec_pitch, spec_roll = df.loc['spectator'].to_numpy()

        # creating transform for spectator
        spectator_transform = carla.Transform(carla.Location(x=spec_x, y=spec_y, z=spec_z),
                                              carla.Rotation(yaw=spec_yaw, pitch=spec_pitch, roll=spec_roll))

        # targeted parking spot 
        down_left_x, down_left_y = df.loc['down_left', 'x':'y'].to_numpy()
        upper_left_x, upper_left_y = df.loc['upper_left', 'x':'y'].to_numpy()
        upper_right_x, upper_right_y = df.loc['upper_right', 'x':'y'].to_numpy()
        down_right_x, down_right_y = df.loc['down_right', 'x':'y'].to_numpy()
        # center of targeted parking spot 
        center_x = (down_left_x + upper_right_x + down_right_x + upper_left_x)/4.0
        center_y = (down_left_y + upper_right_y + down_right_y + upper_left_y)/4.0
        center_rotation = df.loc['center_orientation','yaw'] # in degrees

        # parking spot on left side of targeted parking spot 
        left_down_left_x, left_down_left_y = df.loc['left_down_left', 'x':'y'].to_numpy()
        left_upper_left_x, left_upper_left_y = df.loc['left_upper_left', 'x':'y'].to_numpy()
        left_upper_right_x, left_upper_right_y = df.loc['left_upper_right', 'x':'y'].to_numpy()
        left_down_right_x, left_down_right_y = df.loc['left_down_right', 'x':'y'].to_numpy()

        if np.isnan(left_down_right_x): # this condition is enough for checking if left parking spot is empty and exists or not
            left_parking_spot = None
        else:
            # center of parking spot on left side of targeted parking spot 
            left_center_x = (left_down_left_x + left_upper_left_x + left_upper_right_x + left_down_right_x)/4.0
            left_center_y = (left_down_left_y + left_upper_left_y + left_upper_right_y + left_down_right_y)/4.0
            left_center_rotation = df.loc['left_center_orientation','yaw'] # in degrees

            left_parking_spot = carla.Transform(carla.Location(x=left_center_x, y=left_center_y, z=self.spawning_z_offset),
                                                carla.Rotation(yaw = left_center_rotation))

        # parking spot on right side of targeted parking spot 
        right_down_left_x, right_down_left_y = df.loc['right_down_left', 'x':'y'].to_numpy()
        right_upper_left_x, right_upper_left_y = df.loc['right_upper_left', 'x':'y'].to_numpy()
        right_upper_right_x, right_upper_right_y = df.loc['right_upper_right', 'x':'y'].to_numpy()
        right_down_right_x, right_down_right_y = df.loc['right_down_right', 'x':'y'].to_numpy()

        if np.isnan(right_down_left_x): # this condition is enough for checking if right parking spot is empty and exists or not
            right_parking_spot = None
        else:
            # center of parking spot on right side of targeted parking spot 
            right_center_x = (right_down_left_x + right_upper_left_x + right_upper_right_x + right_down_right_x)/4.0
            right_center_y = (right_down_left_y + right_upper_left_y + right_upper_right_y + right_down_right_y)/4.0
            right_center_rotation = df.loc['right_center_orientation','yaw'] # in degrees

            right_parking_spot = carla.Transform(carla.Location(x=right_center_x, y=right_center_y, z=self.spawning_z_offset),
                                                 carla.Rotation(yaw=right_center_rotation))

        # dictionary for map of important parking spots
        parking_map = { 
                      'down_left'          : carla.Location(x=down_left_x, y=down_left_y),
                      'upper_left'         : carla.Location(x=upper_left_x, y=upper_left_y),
                      'upper_right'        : carla.Location(x=upper_right_x, y=upper_right_y),
                      'down_right'         : carla.Location(x=down_right_x, y=down_right_y), 
                      'center'             : carla.Transform(carla.Location(x=center_x, y=center_y), carla.Rotation(yaw = center_rotation)),
                      'left_parking_spot'  : left_parking_spot,
                      'right_parking_spot' : right_parking_spot
                      }

        return start_transform, spectator_transform, parking_map

    # function for reseting environment and starting new episode
    def reset(self):
        self.collision_hist = []
        self.actor_list = []
 
        # spawning of our agent - Tesla Model 3 
        # self.vehicle = self.world.spawn_actor(self.model_3, self.start_transform) 
        self.vehicle = self.world.spawn_actor(self.model_3, self.start_transform)
        self.actor_list.append(self.vehicle)

        # forcing our agent not to move for 5 seconds -> then we can apply some control
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(0.5)

        # spawning non-moving vehicles on left/right side of targeted parking spot
        if VEHICLES_ON_SIDE_AVAILABLE:
            # spawning redudant agent on left side of our targeted parking spot for better learning
            if self.parking_map['left_parking_spot'] != None:
                self.vehicle_left = self.world.spawn_actor(self.model_3, self.parking_map['left_parking_spot'])
                self.actor_list.append(self.vehicle_left)

                # forcing added vehicle on left side of parking spot not to move
                self.vehicle_left.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

            # spawning redudant agent on right side of our targeted parking spot for better learning
            if self.parking_map['right_parking_spot'] != None:
                self.vehicle_right = self.world.spawn_actor(self.model_3, self.parking_map['right_parking_spot'])
                self.actor_list.append(self.vehicle_right)

                # forcing added vehicle on right side of parking spot not to move
                self.vehicle_right.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # adding collison sensor on our agent
        collision_sensor = self.blueprint_library.find('sensor.other.collision')
        collision_sensor_transform = carla.Transform(carla.Location(x=self.model_3_length/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=0.0))
        self.collision_sensor = self.world.spawn_actor(collision_sensor, collision_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda data: self.collision_data(data))

        # --------------------------------------------------------------------------------------
        # adding 8 radars on 8 positions on vehicle

        # getting radar sensor 
        radar_sensor = self.blueprint_library.find('sensor.other.radar')

        # getting all transforms for all 8 radars
        radar_0_transform = carla.Transform(carla.Location(x=self.model_3_length/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=0.0))
        radar_45_transform = carla.Transform(carla.Location(x=self.model_3_length/2.0, y=self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=45.0))
        radar_90_transform = carla.Transform(carla.Location(y=self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=90.0))
        radar_135_transform = carla.Transform(carla.Location(x=-self.model_3_length/2.0, y=self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=135.0))
        radar_180_transform = carla.Transform(carla.Location(x=-self.model_3_length/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=180.0))
        radar_225_transform = carla.Transform(carla.Location(x=-self.model_3_length/2.0, y=-self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=225.0))
        radar_270_transform = carla.Transform(carla.Location(y=-self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=270.0))
        radar_315_transform = carla.Transform(carla.Location(x=self.model_3_length/2.0, y=-self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=315.0))

        # spawning all 8 radars attached to the vehicle
        self.radar_0 = self.world.spawn_actor(radar_sensor, radar_0_transform, attach_to=self.vehicle)
        self.radar_45 = self.world.spawn_actor(radar_sensor, radar_45_transform, attach_to=self.vehicle)
        self.radar_90 = self.world.spawn_actor(radar_sensor, radar_90_transform, attach_to=self.vehicle)
        self.radar_135 = self.world.spawn_actor(radar_sensor, radar_135_transform, attach_to=self.vehicle)
        self.radar_180 = self.world.spawn_actor(radar_sensor, radar_180_transform, attach_to=self.vehicle)
        self.radar_225 = self.world.spawn_actor(radar_sensor, radar_225_transform, attach_to=self.vehicle)
        self.radar_270 = self.world.spawn_actor(radar_sensor, radar_270_transform, attach_to=self.vehicle)
        self.radar_315 = self.world.spawn_actor(radar_sensor, radar_315_transform, attach_to=self.vehicle)

        # appending all 8 radars to the actor_list
        self.actor_list.append(self.radar_0)
        self.actor_list.append(self.radar_45)
        self.actor_list.append(self.radar_90)
        self.actor_list.append(self.radar_135)
        self.actor_list.append(self.radar_180)
        self.actor_list.append(self.radar_225)
        self.actor_list.append(self.radar_270)
        self.actor_list.append(self.radar_315)

        # radar data listening
        self.radar_0.listen(lambda radar_data: self.radar_data(radar_data, key='radar_0'))
        self.radar_45.listen(lambda radar_data: self.radar_data(radar_data, key='radar_45'))
        self.radar_90.listen(lambda radar_data: self.radar_data(radar_data, key='radar_90'))
        self.radar_135.listen(lambda radar_data: self.radar_data(radar_data, key='radar_135'))
        self.radar_180.listen(lambda radar_data: self.radar_data(radar_data, key='radar_180'))
        self.radar_225.listen(lambda radar_data: self.radar_data(radar_data, key='radar_225'))
        self.radar_270.listen(lambda radar_data: self.radar_data(radar_data, key='radar_270'))
        self.radar_315.listen(lambda radar_data: self.radar_data(radar_data, key='radar_315'))

        # --------------------------------------------------------------------------------------

        # waiting a bit for spawning all sensors
        time.sleep(0.5)

        # starting episode
        self.episode_start = time.time()

        # again, forcing our vehicle not to move until step() makes him move
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # packing first current state after reseting environment
        self.current_state, _ = self.get_current_state()

        return self.current_state

    # function for appending new collision event to collision history
    def collision_data(self, data):
        self.collision_hist.append(data)

    # function for processing radar readings
    def radar_data(self, radar_data, key):

        # numpy array with size (len(radar_data), 4) with values like --> [[vel, altitude, azimuth, depth],...[,,,]]:
        radar_points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')).reshape((len(radar_data), 4))

        # getting only min depth
        min_depth_radar_reading = min(np.reshape(radar_points[:,3],(len(radar_data),)))

        # load closest radar reading 
        self.radar_readings[key] = min_depth_radar_reading

    # function for getting angles from 0 to 30 degs
    def tranfsorm_angle(self, angle):
        angle_360 = 360 + angle if angle < 0 else angle
        return angle_360

    # function for getting current state of Carla agent
    def get_current_state(self):

        # current state of the agents is defined as one vector with these values (normalized):
        # 8 radar readings - min depth of radar readings form 8 different positions on vehicle, nomalized by 20
        # x_rel - difference of x coordinate center of goal parking spot and of center of vehicle in global coordinates (relative x)
        # y_rel - difference of y coordinate center of goal parking spot and of center of vehicle in global coordinates (relative y)
        # angle - angle of rotation along axial axis (z-axis) (relative angle)
        # vx - linear velocity along x-axis of vehicle
        # ax - acceleration along x-axis of vehicle
        # wz - angular velocity along z-axis of vehicle - rotation velocity
        # distance_to_goal - Euclidian distance from current position of vehicle to the goal position  

        current_vehicle_transform = self.vehicle.get_transform()
        current_vehicle_location = current_vehicle_transform.location
        current_vehicle_rotation = current_vehicle_transform.rotation

        current_vehicle_linear_velocity = self.vehicle.get_velocity()
        current_vehicle_angular_velocity = self.vehicle.get_angular_velocity()

        current_vehicle_acceleration = self.vehicle.get_acceleration()

        # defining states --> in this config (Akerman's drive) there is no vy, only vx and wz
        x_rel = (self.parking_map['center'].location.x - current_vehicle_location.x)/40.0  
        y_rel = (self.parking_map['center'].location.y - current_vehicle_location.y)/40.0
        angle = (self.tranfsorm_angle(current_vehicle_rotation.yaw))/360
        vx = (current_vehicle_linear_velocity.x)/50.0  
        ax = (current_vehicle_acceleration.x)/50.0 
        wz = (current_vehicle_angular_velocity.z)/50.0   
        distance_to_goal = (current_vehicle_location.distance(self.parking_map['center'].location))/40.0

        # definition of current state - concatenation of 8 radar readings and other values
        current_state = [radar_reading/20.0 for radar_reading in list(self.radar_readings.values())] + [x_rel, y_rel, angle, vx, ax, wz, distance_to_goal]

        # dictionary for current state (sensor that are not radar readings)
        sensor_values_dict = {
                                'x': x_rel,
                                'y': y_rel,
                                'angle': angle,
                                'vx': vx,
                                'ax': ax,
                                'wz': wz,
                                'distance_to_goal': distance_to_goal,
                             }

        # concatenation of radar_readings and other sensors readings and constructing new (current_state_dict) dictionary of them
        current_state_dict = self.radar_readings.copy()
        current_state_dict.update(sensor_values_dict)

        current_state = np.array(current_state, dtype='float32').reshape((STATE_SIZE,)) # 1D array of current state 

        return current_state, current_state_dict

    # function for taking actions and getting reward for that actions
    # this function should be the key function a this should return new state, reward and done flag (potentially some info)
    def step(self, actions):

        # actions are defined as dictionary with 3 values for throttle, brake and steer
        # actions = {
        #             'throttle': value_throttle, 
        #             'steer': value_steer,
        #             'reverse': value_reverse (True/False),
        #           }

        reverse = False if actions['throttle'] >= 0 else True
        throttle = abs(actions['throttle'])
        steer = actions['steer']

        # applying control (calculated from neural network) on our vehicle
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, reverse=reverse))

        # waiting some minor time, so that control can be properly applied
        time.sleep(0.5)

        # ---------------- REWARD CALCULATION ----------------

        # first, get new current state, because it is been changed due to applied control
        self.current_state, current_state_dict = self.get_current_state()

        if len(self.collision_hist) != 0: 
        # if there was any collisons reward is very bad
        # and we are done for this episode
            done = True
            reward = -200

        elif current_state_dict['distance_to_goal'] > 25:
        # if we are more 25 meters away from the specified parking spot
        # then we are giving it some bad reward, and setting done to True 
        # for this episode, because vehicle is far away
            done = True
            reward = -50

        elif (self.episode_start + NON_MOVING_SECONDS) < time.time(): 
        # potentially there could be problem if our agent feels like it is better
        # for him not to move, then seconds per episode will be broken, and we have to
        # penalize him for that... this could be wrapped with SECONDS_PER_EPISODE but because 
        # then this condition would also break when episode finishes, it is now separated

            loc = carla.Location(x=current_state_dict['x'], y=current_state_dict['y'])
            traveled_distance = self.start_transform.location.distance(loc)

            if traveled_distance <= 2:
                done = True
                reward = -20
            else:
                done = False
                reward = self.calculate_reward(current_state_dict['distance_to_goal'], current_state_dict['angle'])

        else:
        # in every other situation reward will be calculated as proposed
            done = False
            reward = self.calculate_reward(current_state_dict['distance_to_goal'], current_state_dict['angle'])

        if ((self.episode_start + SECONDS_PER_EPISODE) < time.time()):
            # if we exceeded time limit for episode, we are breaking that episode
            done = True

        return self.current_state, reward, done

    # function for calculating current reward for just taken action
    def calculate_reward(self, distance, angle):

        # calculating realtive angle to the goal orientation
        theta = self.tranfsorm_angle(self.parking_map['center'].rotation.yaw) - angle

        # clipping distance so that there wont be +inf 
        if distance < 0.001:
            distance = 0.001

        # reward value calculation
        reward = (1.0/distance) * abs(np.cos(np.deg2rad(theta)))

        return reward

    # function for destroying all Carla actors
    def destroy_actors(self):

        # if actor_list is not empty, then we are 
        # deleting all actors
        if self.actor_list:
            for actor in self.actor_list:
                    success = actor.destroy()

# ---------------------------------------------------------------------------------------------------
# ORNSTEIN-UHLENBECK PROCESS NOISE CLASS
# ---------------------------------------------------------------------------------------------------
# for implementing better exploration by the Actor network it is nice to add some noisy perturabtions
# this process samples noise from a correlated normal distribution.
class OUActionNoise:

    # contructor of the class
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    # function for calling to generate noise using OU noise formula
    def __call__(self):

        # this formula is taken from ----> https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        # store x into x_prev
        # makes next noise dependent on current one (as its said, this is time correlated stochastic process)
        self.x_prev = x
        return x

    # function for reseting samples of OU noise
    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mu)

    # function for sampling noise 
    def sample_noise(self, x):

        noise = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.mu.shape)

        return noise
        
# ---------------------------------------------------------------------------------------------------
# DEEP DETERMINISTIC POLICY GRADIENT (DDPG) AGENT CLASS
# ---------------------------------------------------------------------------------------------------
# class serves for DDPG reinforcement learning problem solving; 
# it uses two dependent nets - Actor & Critic and gets trained using DDPG algorithm
# and using Target models for both of these 2 NNs and Experience Replay from the Buffer class
class DDPGAgent:

    # constructor of the class -> does nothing
    def __init__(self):
        pass

    # function for constucting Actor Model
    def get_actor(self, model_name='', manually_stopped = False):

        # if manually stopped model is that what we want to get
        manual = '_manually_stopped' if manually_stopped else ''

        # getting model name
        model_weights_file_name = 'models/parking_agent_actor' + model_name + manual + '.h5'

        # initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        # defining inputs
        inputs = layers.Input(shape=(STATE_SIZE,))

        # defining two fully conected layers
        out = layers.Dense(300, activation='relu')(inputs)
        out = layers.Dense(600, activation='relu')(out)

        # # output layers for 2 actions
        # throttle_action = layers.Dense(1, activation='sigmoid', kernel_initializer=last_init)(out)
        # brake_action = layers.Dense(1, activation='sigmoid', kernel_initializer=last_init)(out)
        # steer_action = layers.Dense(1, activation='tanh', kernel_initializer=last_init)(out)

        # # concatenated output layers
        # outputs = layers.Concatenate()([throttle_action, brake_action, steer_action])

        outputs = layers.Dense(2, activation='tanh', kernel_initializer=last_init)(out)

        # defining Actor NN model
        model = tf.keras.Model(inputs, outputs)

        # loading weights if they exist
        if LOAD_MODEL_WEIGHTS_ENABLED == True and os.path.exists(model_weights_file_name):
            model.load_weights(model_weights_file_name)

        # if they don't exist, and it is about target nn (this means that we are creating new actor model), then copy actor weights
        elif model_name == '_target':
            model.set_weights(actor_model.get_weights())

        return model

    # function for constucting Critic Model
    def get_critic(self, model_name='', manually_stopped = False):

        # if manually stopped model is that what we want to get
        manual = '_manually_stopped' if manually_stopped else ''

        # getting model name
        model_weights_file_name = 'models/parking_agent_critic' + model_name + manual + '.h5'

        # state as input
        state_input = layers.Input(shape=(STATE_SIZE,))
        state_out = layers.Dense(100, activation='relu')(state_input)
        state_out = layers.Dense(200, activation='relu')(state_out)

        # action as input
        action_input = layers.Input(shape=(ACTIONS_SIZE,))
        action_out = layers.Dense(200, activation='relu')(action_input)

        # both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(300, activation='relu')(concat)
        out = layers.Dense(600, activation='relu')(out)
        outputs = layers.Dense(1)(out)

        # defining Critic NN model
        # outputs single value for give state-action (outputs critic)
        model = tf.keras.Model([state_input, action_input], outputs)

        # loading weights if they exist
        if LOAD_MODEL_WEIGHTS_ENABLED == True and os.path.exists(model_weights_file_name):
            model.load_weights(model_weights_file_name)

        # if they don't exist, and it is about target nn (this means that we are creating new critic model), then copy critic weights
        elif model_name == '_target':
            model.set_weights(critic_model.get_weights())

        return model

    # getting policy of actor model for the given state (+ added noise)
    def policy(self, state, noise_objects_dict):

        global epsilon

        # sampling action from Actor model
        sampled_actions = tf.squeeze(actor_model(state))

        # # adding noise to action
        noise_throttle = noise_objects_dict['throttle']
        noise_steer= noise_objects_dict['steer']

        sampled_actions = sampled_actions.numpy()

        # exploration amount
        epsilon -= 1.0/EXPLORE

        # unpacking sampled actions
        # throttle = float(sampled_actions[0] + max(epsilon, 0)*noise_throttle())
        # steer = float(sampled_actions[1] + max(epsilon, 0)*noise_steer())

        throttle = float(sampled_actions[0] + max(epsilon, 0)*noise_throttle.sample_noise(sampled_actions[0]))
        steer = float(sampled_actions[1] + max(epsilon, 0)*noise_steer.sample_noise(sampled_actions[1]))

        if throttle > 1:
            throttle = 1
        elif throttle < -1:
            throttle = -1

        if steer > 1:
            steer = 1
        elif steer < -1:
            steer = -1

        # packing action in array
        legal_actions_array = np.array([throttle, steer], dtype='float32').reshape((ACTIONS_SIZE,))

        # packing legal action in dictionary
        legal_actions_dict = {
                              'throttle': throttle,
                              'steer': steer
                             }

        return legal_actions_array, legal_actions_dict

# ---------------------------------------------------------------------------------------------------
# REPLAY BUFFER CLASS
# ---------------------------------------------------------------------------------------------------
# class implements so called Experience Replay
#
# Critic loss - Mean Squared Error of 'y - Q(s, a)' where 'y' is the expected return as seen by the Target network,
# and 'Q(s, a)' is action value predicted by the Critic network
# 'y' is a moving target that the critic model tries to achieve; we make this target stable by updating the Target model slowly
#
# Actor loss - This is computed using the mean of the value given by the Critic network for the actions taken by the Actor network
# We seek to maximize this quantity
# Hence we update the Actor network so that it produces actions that get the maximum predicted value as seen by the Critic, for a given state.
class ReplayBuffer:

    # constructor of the class
    def __init__(self, buffer_capacity=100000, batch_size=64):

        # number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity

        # num of tuples to train on.
        self.batch_size = batch_size

        # it tells us number of times record() was called.
        self.buffer_counter = 0

        # forming buffer memory 
        self.state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))
        self.action_buffer = np.zeros((self.buffer_capacity, ACTIONS_SIZE))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    # function for recording experience - it takes [s,a,r,s',d] obervation as input
    # [s,a,r,s',d] - stands for [state, action, reward for that action, new state, done]
    def record(self, observation):

        # set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = observation['previous_state']
        self.action_buffer[index] = observation['actions']
        self.reward_buffer[index] = observation['reward']
        self.next_state_buffer[index] = observation['next_state']
        self.done_buffer[index] = observation['done']

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function # eager execution
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):

        # training and updating Actor & Critic networks (as said in the begining of the class implementation)
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + GAMMA * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)

            # using '-value' as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    # computing the loss and updating parameters
    def learn(self):

        # get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)

        # randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # conversion to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# ---------------------------------------------------------------------------------------------------
# FUNCTION FOR SLOWLY UPDATING WEIGHTS ON TARGET MODELS
# ---------------------------------------------------------------------------------------------------
# this function updates target parameters slowly, based on 'tau' rate
# this is very useful for stability of algorithm (developed by DeepMind, as a upgrade of Q-learning)

@tf.function # eager execution
def update_target(target_weights, actual_weights):
    for (a, b) in zip(target_weights, actual_weights):
        a.assign(b * TAU + a * (1 - TAU))

# ---------------------------------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------------------------------
# main program for training RL parking agent using Deep Deterministic Policy Gradient (DDPG) agent
if __name__ == '__main__':

    # seeding for more repetitive results
    random.seed(1)
    np.random.seed(2)
    tf.random.set_seed(3)

    # memory fraction, mostly used when training multiple agents
    # GPU_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # set_session(tf.Session(config=tf.ConfigProto(gpu_options=GPU_options)))

    # creating models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    try: 

        # default spawning/starting point
        starting_waypoint = 'entrance'

        waypoint_selected = False   

        # loop for waiting for user response on which spawning point to spawn vehicle on start
        while(True):

            # selected waypoint for start position
            selected_waypoint = input('Select waypoint for starting position (e/i): ')

            if selected_waypoint == '':
                continue

            elif selected_waypoint == 'e':
                starting_waypoint = 'entrance'
                waypoint_selected = True

            elif selected_waypoint == 'i':
                starting_waypoint = 'intersection'
                waypoint_selected = True

            if waypoint_selected == True:
                break

            time.sleep(_SLEEP_TIME_)

        # creating Carla environment object
        env = CarlaEnvironment(starting_waypoint)

        # creating agent and environment
        agent = DDPGAgent()

        # Ornstein-Uhlenbeck noise objects
        ou_noise_throttle = OUActionNoise(mu=np.zeros(1), sigma=0.2*np.ones(1), theta=1.0)
        ou_noise_steer = OUActionNoise(mu=np.zeros(1), sigma=0.1*np.ones(1), theta=0.6)

        ou_noise_dict = {
                          'throttle': ou_noise_throttle,
                          'steer': ou_noise_steer
                         }

        # making Actor and Critic neural networks
        actor_model = agent.get_actor(manually_stopped=False)
        critic_model = agent.get_critic(manually_stopped=False)

        # making target Actor and Critic neural networks
        target_actor = agent.get_actor(model_name='_target', manually_stopped=False)
        target_critic = agent.get_critic(model_name='_target', manually_stopped=False)

        # defining optimizers for actor and critic NNs
        actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)
        critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)

        # creating replay buffer
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY, BATCH_SIZE)

        # to store reward history of each episode
        episode_reward_list = []

        # to store average reward history of last few episodes
        average_reward_list = []

        # steps taken for episodes
        steps_list = []

        print('-----------------Start of training process---------------')

        # main loop for training 
        for episode in range(1,TOTAL_EPISODES+1):

            # defining first state over one episode
            previous_state = env.reset()

            # reset episodic reward
            episodic_reward = 0

            # reset episodic inner loop counter
            step = 0

            # while loop for iterating in one episode
            while True:

                # incrementing inner loop counter
                step += 1

                # tensorlow previous state
                tf_previous_state = tf.expand_dims(tf.convert_to_tensor(previous_state), 0)

                # sampling actions
                actions_arr, actions_dict = agent.policy(tf_previous_state, ou_noise_dict)

                print('Throttle: {:.3f}, Steer: {:.3f}'.format(actions_dict['throttle'], actions_dict['steer']))

                # recieve state and reward from environment
                next_state, reward, done = env.step(actions_dict)

                # packing of observation
                observation = {
                                'previous_state': previous_state,
                                'actions': actions_arr,
                                'reward': reward,
                                'next_state': next_state,
                                'done': done
                              }

                # writing down records from previous state to this new state
                replay_buffer.record(observation)

                # recording reward
                episodic_reward += reward

                # learning process
                replay_buffer.learn()

                # updating target models, slowly
                update_target(target_actor.variables, actor_model.variables)
                update_target(target_critic.variables, critic_model.variables)

                # this episode ends if done is True
                if done:
                    break

                # transition for previous state
                previous_state = next_state

            # destroying actors after one episode
            env.destroy_actors()

            # on the end of one episode, append to the list for episodic rewards, that episodic reward
            episode_reward_list.append(episodic_reward)

            print('Episode * {} * Average Episode Reward is ==> {}'.format(episode, episodic_reward/step))

            # mean of last AVERAGE_EPISODES_COUNT episodes is average_reward
            average_reward = np.mean(episode_reward_list[-AVERAGE_EPISODES_COUNT:])
            average_reward_list.append(average_reward)

            steps_list.append(step)

        print('-----------------End of training process---------------')
        
        # if training proceeds correctly, the average episodic reward should increase with time.
        # change learning rates, 'tau' valuee, and architectures for the Actor and Critic networks to get better results

        # saving models
        actor_model.save_weights('models/parking_agent_actor.h5')
        critic_model.save_weights('models/parking_agent_critic.h5')

        target_actor.save_weights('models/parking_agent_actor_target.h5')
        target_critic.save_weights('models/parking_agent_critic_target.h5')

        # plotting episodic and average episodic reward
        plt.figure(figsize = (10,10), dpi = 100)
        plt.plot(np.arange(1,TOTAL_EPISODES+1), episode_reward_list, color='red', linewidth=1.2, label='episodic')
        plt.plot(np.arange(1,TOTAL_EPISODES+1), average_reward_list, color='green', linewidth=1.2, label='average episodic')
        plt.xlabel('Episode')
        plt.ylabel('Rt')
        plt.grid()
        plt.title('Episodic and Average Episodic Reward \n (average over every last {} episodes)'.format(AVERAGE_EPISODES_COUNT))
        plt.legend(loc='upper right')
        plt.savefig(FOLDER_PATH + '/training_pictures_and_video/training_rewards_'+ str(time.time()) +'.png')

        # plotting number of steps taken in episode
        plt.figure(figsize = (10,10), dpi = 100)
        plt.plot(np.arange(1,TOTAL_EPISODES+1), steps_list, color='green', linewidth=1.2)
        plt.xlabel('Episode')
        plt.ylabel('Number of steps')
        plt.grid()
        plt.title('Number of steps taken over episodes \n (episode lasts {} seconds)'.format(SECONDS_PER_EPISODE))
        plt.savefig(FOLDER_PATH + '/training_pictures_and_video/training_steps_'+ str(time.time()) +'.png')

    # if program is manually stopped (using Ctrl + C) then models are stored as _manually_stopped 
    except KeyboardInterrupt:

        # saving manually stopped models
        actor_model.save_weights('models/parking_agent_actor_manually_stopped.h5')
        critic_model.save_weights('models/parking_agent_critic_manually_stopped.h5')

        target_actor.save_weights('models/parking_agent_actor_target_manually_stopped.h5')
        target_critic.save_weights('models/parking_agent_critic_target_manually_stopped.h5')

        # plotting episodic and average episodic reward, even thought it is early/manually stopped
        plt.figure(figsize = (10,10), dpi = 100)
        plt.plot(np.arange(1, len(episode_reward_list)+1), episode_reward_list, color='red', linewidth=1.2, label='episodic')
        plt.plot(np.arange(1, len(average_reward_list)+1), average_reward_list, color='green', linewidth=1.2, label='average episodic')
        plt.xlabel('Episode')
        plt.ylabel('Rt')
        plt.grid()
        plt.title('Episodic and Average Episodic Reward \n (averaged over every last {} episodes) \n --- manually stopped at episode {}/{} ---'.format(AVERAGE_EPISODES_COUNT, episode-1, TOTAL_EPISODES))
        plt.legend(loc='upper right')
        plt.savefig(FOLDER_PATH + '/training_pictures_and_video/training_rewards_'+ str(time.time()) +'.png')

        # plotting number of steps taken in episode
        plt.figure(figsize = (10,10), dpi = 100)
        plt.plot(np.arange(1, len(steps_list)+1), steps_list, color='green', linewidth=1.2)
        plt.xlabel('Episode')
        plt.ylabel('Number of steps')
        plt.grid()
        plt.title('Number of steps taken over episodes \n (episode lasts {} seconds) \n --- manually stopped at episode {}/{} ---'.format(SECONDS_PER_EPISODE, episode-1, TOTAL_EPISODES))
        plt.savefig(FOLDER_PATH + '/training_pictures_and_video/training_steps_'+ str(time.time()) +'.png')

   