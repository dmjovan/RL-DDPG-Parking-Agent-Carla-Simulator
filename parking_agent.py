# ---------------------------------------------------------------------------------------------------
# IMPORTING ALL NECESSARY LIBRARIES
# ---------------------------------------------------------------------------------------------------
import glob
import os
import sys
import random
import time
import numpy as np
import math
import pandas as pd
from datetime import datetime
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
_HOST_ = '127.0.0.1'
_PORT_ = 2000
_SLEEP_TIME_ = 0.5 

FOLDER_PATH = os.getcwd()
MAP_CSV_PATH = FOLDER_PATH + '/parking_map.csv'

VEHICLES_ON_SIDE_AVAILABLE = False

ACTIONS_SIZE = 2
STATE_SIZE = 16

MAX_COLLISION_IMPULSE = 50
MAX_DISTANCE = 20
MAX_REWARD = 1.0
SIGMA = 7.0

# ---------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES FOR TRAINING 
# ---------------------------------------------------------------------------------------------------
LOAD_MODEL_WEIGHTS_ENABLED = False

MEMORY_FRACTION = 0.3333

TOTAL_EPISODES = 10000
STEPS_PER_EPISODE = 1000
AVERAGE_EPISODES_COUNT = 40

NON_MOVING_STEPS = 100

REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 64

CRITIC_LR = 0.001
ACTOR_LR = 0.0001
GAMMA = 0.99
TAU = 0.001

epsilon = 1
EXPLORE = 100000.0
MIN_EPSILON = 0.000001

# ---------------------------------------------------------------------------------------------------
# CARLA ENVIRONMENT CLASS
# ---------------------------------------------------------------------------------------------------
class CarlaEnvironment:

    def __init__(self):

        self.client = carla.Client(_HOST_, _PORT_)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town05') 
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        self.model_3 = self.blueprint_library.filter('model3')[0]

        # https://www.automobiledimension.com/photos/tesla-model-3-2018.jpg <-- image with Tesla Model 3 dimensions
        self.model_3_heigth = 1.443 
        self.model_3_length = 4.694 
        self.model_3_width = 2.089 

        self.spawning_z_offset = 0.5

        self.parking_map, spectator_transform = self.get_parking_map()

        self.world.get_spectator().set_transform(spectator_transform)

        self.draw_goal()

        self.radar_readings = {
                                'radar_0'  : 100.0,
                                'radar_45' : 100.0,
                                'radar_90' : 100.0,
                                'radar_135': 100.0,
                                'radar_180': 100.0,
                                'radar_225': 100.0,
                                'radar_270': 100.0, 
                                'radar_315': 100.0 
                               }

    def get_parking_map(self):

        """
        Function for getting real spots on parking lot from previously catched 
        data while observing environment through get_parking_map.py script.

        :params: 
            None

        :return:
            - parking_map: dictionary with 3 specific carla.Transform objects for
                           goal, left of the goal and right of the goal parking spot

            - spectator_transform: carla.Transform object with location and rotation of 
                                   spectator camera
        """

        df = pd.read_csv(MAP_CSV_PATH, index_col = ['position'])
        df = df.apply(pd.to_numeric, errors='coerce')

        # --------------------------- GOAL PARKING SPOT -----------------------------------

        goal_down_left_x, goal_down_left_y = df.loc['goal_down_left', 'x':'y'].to_numpy()
        goal_upper_left_x, goal_upper_left_y = df.loc['goal_upper_left', 'x':'y'].to_numpy()
        goal_upper_right_x, goal_upper_right_y = df.loc['goal_upper_right', 'x':'y'].to_numpy()
        goal_down_right_x, goal_down_right_y = df.loc['goal_down_right', 'x':'y'].to_numpy()

        # center of target parking spot 
        goal_center_x = (goal_down_left_x + goal_upper_left_x + goal_upper_right_x + goal_down_right_x)/4.0
        goal_center_y = (goal_down_left_y + goal_upper_left_y + goal_upper_right_y + goal_down_right_y)/4.0
        goal_rotation = df.loc['goal_orientation','yaw'] # in degrees

        goal_parking_spot = carla.Transform(carla.Location(x=goal_center_x, y=goal_center_y), carla.Rotation(yaw=goal_rotation))

        # --------------------------- PARKING SPOT ON LEFT -----------------------------------

        left_down_left_x, left_down_left_y = df.loc['left_down_left', 'x':'y'].to_numpy()
        left_upper_left_x, left_upper_left_y = df.loc['left_upper_left', 'x':'y'].to_numpy()
        left_upper_right_x, left_upper_right_y = df.loc['left_upper_right', 'x':'y'].to_numpy()
        left_down_right_x, left_down_right_y = df.loc['left_down_right', 'x':'y'].to_numpy()

        if np.isnan(left_down_right_x): 
            left_parking_spot = None
        else:
            # center of parking spot on left side of target parking spot 
            left_center_x = (left_down_left_x + left_upper_left_x + left_upper_right_x + left_down_right_x)/4.0
            left_center_y = (left_down_left_y + left_upper_left_y + left_upper_right_y + left_down_right_y)/4.0
            left_rotation = df.loc['left_orientation','yaw']

            left_parking_spot = carla.Transform(carla.Location(x=left_center_x, y=left_center_y, z=self.spawning_z_offset),
                                                carla.Rotation(yaw = left_rotation))

        # --------------------------- PARKING SPOT ON RIGHT ---------------------------------

        right_down_left_x, right_down_left_y = df.loc['right_down_left', 'x':'y'].to_numpy()
        right_upper_left_x, right_upper_left_y = df.loc['right_upper_left', 'x':'y'].to_numpy()
        right_upper_right_x, right_upper_right_y = df.loc['right_upper_right', 'x':'y'].to_numpy()
        right_down_right_x, right_down_right_y = df.loc['right_down_right', 'x':'y'].to_numpy()

        if np.isnan(right_down_left_x):
            right_parking_spot = None
        else:
            # center of parking spot on right side of target parking spot 
            right_center_x = (right_down_left_x + right_upper_left_x + right_upper_right_x + right_down_right_x)/4.0
            right_center_y = (right_down_left_y + right_upper_left_y + right_upper_right_y + right_down_right_y)/4.0
            right_rotation = df.loc['right_orientation','yaw'] 

            right_parking_spot = carla.Transform(carla.Location(x=right_center_x, y=right_center_y, z=self.spawning_z_offset),
                                                 carla.Rotation(yaw=right_rotation))

        # --------------------------- SPECTATOR CAMERA TRANSFORM ------------------------------

        spec_x, spec_y, spec_z, spec_yaw, spec_pitch, spec_roll = df.loc['spectator'].to_numpy()

        spectator_transform = carla.Transform(carla.Location(x=spec_x, y=spec_y, z=spec_z),
                                              carla.Rotation(yaw=spec_yaw, pitch=spec_pitch, roll=spec_roll))

        # --------------------------- PARKING MAP DICTIONARY ----------------------------------
        parking_map = { 
                       'goal_down_left'     : carla.Location(x=goal_down_left_x, y=goal_down_left_y, z=0.2),
                       'goal_upper_left'    : carla.Location(x=goal_upper_left_x, y=goal_upper_left_y, z=0.2),
                       'goal_upper_right'   : carla.Location(x=goal_upper_right_x, y=goal_upper_right_y, z=0.2),
                       'goal_down_right'    : carla.Location(x=goal_down_right_x, y=goal_down_right_y, z=0.2),
                       'goal_parking_spot'  : goal_parking_spot,
                       'left_parking_spot'  : left_parking_spot,
                       'right_parking_spot' : right_parking_spot
                      }

        return parking_map, spectator_transform

    def random_spawn(self, mode):

        """
        Function for random spawning on places near parking. 

        :params:
            - mode: 3 modes are currently provided:
                    - carla_recommended: spawn points near parking recommended by Carla authors
                    - random_lane: spawn points in lane closest to the parking
                    - random_entrance: spawn points in spatial rectangle in the entrance
        :return:
            - spawn_transform: carla.Transform object for final spawn position

        """

        # --------------------------- PREPROCESSING FOR CARLA RECOMMENDED SPAWN POINTS ------------------------------
        x0_carla, y0_carla = 0.0, 0.0

        if mode =='carla_recommended': 
            x0_carla, y0_carla = self.get_carla_recommended_spawn_points(self, x_min=-1, x_max=36, y_min=-49, y_max=-10)
            
        # --------------------------- DICTIONARY OF CHARACTERISTIC VALUES FOR EACH MODE ------------------------------
        mode_values_dict = {
                             'carla_recommended': {
                                                    'x0': x0_carla,
                                                    'y0': y0_carla,
                                                    'x_min': -2,
                                                    'x_max': 2,
                                                    'y_min': -2,
                                                    'y_max': 2,
                                                    'yaw_min': -180,
                                                    'yaw_max': 180
                                                  },
                             'random_lane':       {
                                                    'x0': 0.0,
                                                    'y0': 0.0,
                                                    'x_min': 23.5,
                                                    'x_max': 30,
                                                    'y_min': -44,
                                                    'y_max': -15,
                                                    'yaw_min': 150,
                                                    'yaw_max': 210
                                                  },
                             'random_entrance':   {
                                                    'x0': 0.0,
                                                    'y0': 0.0,
                                                    'x_min': 13,
                                                    'x_max': 29,
                                                    'y_min': -33,
                                                    'y_max': -27,
                                                    'yaw_min': 150,
                                                    'yaw_max': 210
                                                  }
                            }

        spawn_transform = self.get_spawn_transform(mode_values_dict[mode])

        return spawn_transform

    def get_carla_recommended_spawn_points(self, x_min, x_max, y_min, y_max):

        """
        Function for generating Carla recommended spawn point in provided 
        coordinate ranges.
            
        :params:
            - x_min: minimum of global x coordinate
            - x_max: maximum of global x coordinate
            - y_min: minimum of global y coordinate
            - y_max: maximum of global y coordinate

        :return:
            - spawn_x: x coordinate of choosen spawn point
            - spawn_y: y coordinate of choosen spawn point

        """

        spawn_points = self.map.get_spawn_points()

        valid_spawn_points = []

        for spawn_point in spawn_points:
            x = spawn_point.location.x
            y = spawn_point.location.y

            if (x >= x_min and x <= x_max) and (y >= y_min and y <= y_max):
                valid_spawn_points.append(spawn_point)

        spawn_location = (random.choice(valid_spawn_points)).location
        spawn_x = spawn_location.x
        spawn_y = spawn_location.y

        return spawn_x, spawn_y

    def get_spawn_transform(self, values):

        """
        Function for generating random spawn transform for vehicle
        depending of input values.
            
        :params:
            - values: dictionary with coordintes of corners for spawning and initial offsets

        :return:
            - spawn_transform: carla.Transform object for spawning location and rotation

        """

        x0 = values['x0']
        y0 = values['y0']

        x_min = values['x_min']
        x_max = values['x_max']

        y_min = values['y_min']
        y_max = values['y_max']

        yaw_min = values['yaw_min']
        yaw_max = values['yaw_max']

        x_random_value = random.random()
        x_random_spawn = x_min + x_random_value*(x_max-x_min)

        y_random_value = random.random()
        y_random_spawn = y_min + y_random_value*(y_max-y_min)

        yaw_random_value = random.random()
        yaw_random_spawn = yaw_min + yaw_random_value*(yaw_max-yaw_min)

        spawn_transform = carla.Transform(carla.Location(x=x_random_spawn+x0, y=y_random_spawn+y0, z=self.spawning_z_offset), carla.Rotation(yaw=yaw_random_spawn))

        return spawn_transform

    def draw_goal(self):

        """
        Function for drawing rectangle on goal parking spot.
            
        :params:
            None

        :return:
            None

        """

        debug = self.world.debug

        begin_1 = self.parking_map['goal_down_left']
        end_1 = self.parking_map['goal_upper_left']

        begin_2 = self.parking_map['goal_upper_left']
        end_2 = self.parking_map['goal_upper_right']

        begin_3 = self.parking_map['goal_upper_right']
        end_3 = self.parking_map['goal_down_right']

        begin_4 = self.parking_map['goal_down_right']
        end_4 = self.parking_map['goal_down_left']

        debug.draw_line(begin_1, end_1, thickness=0.2, color=carla.Color(255,0,0), life_time=0)
        debug.draw_line(begin_2, end_2, thickness=0.2, color=carla.Color(255,0,0), life_time=0)
        debug.draw_line(begin_3, end_3, thickness=0.2, color=carla.Color(255,0,0), life_time=0)
        debug.draw_line(begin_4, end_4, thickness=0.2, color=carla.Color(255,0,0), life_time=0)

    def reset(self):

        """
        Function for reseting environment and starting new episode.
            
        :params:
            None

        :return:
            - current_state: numpy array with shape (STATE_SIZE, ) 
                             with all sensor readings on start of the new episode

        """

        self.collision_impulse = None
        self.last_collision_impulse = None

        self.actor_list = []
 
        # ------------------------------ SPAWNING AGENT ----------------------------------

        spawn_point = self.random_spawn('random_entrance')

        # spawn_point = carla.Transform(carla.Location(x=17.2, y=-29.7, z=self.spawning_z_offset), carla.Rotation(yaw=180.0))

        self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
        self.actor_list.append(self.vehicle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(_SLEEP_TIME_)

        # ------------------------------ SPAWNING OTHER NON-MOVING VEHICLES ----------------------------------
        if VEHICLES_ON_SIDE_AVAILABLE:

            if self.parking_map['left_parking_spot'] != None:
                self.vehicle_left = self.world.spawn_actor(self.model_3, self.parking_map['left_parking_spot'])
                self.actor_list.append(self.vehicle_left)
                self.vehicle_left.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

            if self.parking_map['right_parking_spot'] != None:
                self.vehicle_right = self.world.spawn_actor(self.model_3, self.parking_map['right_parking_spot'])
                self.actor_list.append(self.vehicle_right)
                self.vehicle_right.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # ------------------------------ COLLISION SENSOR  ----------------------------------

        collision_sensor = self.blueprint_library.find('sensor.other.collision')
        collision_sensor_transform = carla.Transform(carla.Location(x=self.model_3_length/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=0.0))
        self.collision_sensor = self.world.spawn_actor(collision_sensor, collision_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda data: self.collision_data(data))

        # ------------------------------ RADARS ON 8 DIFFERENT POSITION ON VEHICLE ---------------------------------- 

        radar_sensor = self.blueprint_library.find('sensor.other.radar')

        radar_0_transform = carla.Transform(carla.Location(x=self.model_3_length/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=0.0))
        radar_45_transform = carla.Transform(carla.Location(x=self.model_3_length/2.0, y=self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=45.0))
        radar_90_transform = carla.Transform(carla.Location(y=self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=90.0))
        radar_135_transform = carla.Transform(carla.Location(x=-self.model_3_length/2.0, y=self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=135.0))
        radar_180_transform = carla.Transform(carla.Location(x=-self.model_3_length/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=180.0))
        radar_225_transform = carla.Transform(carla.Location(x=-self.model_3_length/2.0, y=-self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=225.0))
        radar_270_transform = carla.Transform(carla.Location(y=-self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=270.0))
        radar_315_transform = carla.Transform(carla.Location(x=self.model_3_length/2.0, y=-self.model_3_width/2.0, z=self.model_3_heigth/2.0), carla.Rotation(yaw=315.0))

        self.radar_0 = self.world.spawn_actor(radar_sensor, radar_0_transform, attach_to=self.vehicle)
        self.radar_45 = self.world.spawn_actor(radar_sensor, radar_45_transform, attach_to=self.vehicle)
        self.radar_90 = self.world.spawn_actor(radar_sensor, radar_90_transform, attach_to=self.vehicle)
        self.radar_135 = self.world.spawn_actor(radar_sensor, radar_135_transform, attach_to=self.vehicle)
        self.radar_180 = self.world.spawn_actor(radar_sensor, radar_180_transform, attach_to=self.vehicle)
        self.radar_225 = self.world.spawn_actor(radar_sensor, radar_225_transform, attach_to=self.vehicle)
        self.radar_270 = self.world.spawn_actor(radar_sensor, radar_270_transform, attach_to=self.vehicle)
        self.radar_315 = self.world.spawn_actor(radar_sensor, radar_315_transform, attach_to=self.vehicle)

        self.actor_list.append(self.radar_0)
        self.actor_list.append(self.radar_45)
        self.actor_list.append(self.radar_90)
        self.actor_list.append(self.radar_135)
        self.actor_list.append(self.radar_180)
        self.actor_list.append(self.radar_225)
        self.actor_list.append(self.radar_270)
        self.actor_list.append(self.radar_315)

        self.radar_0.listen(lambda radar_data: self.radar_data(radar_data, key='radar_0'))
        self.radar_45.listen(lambda radar_data: self.radar_data(radar_data, key='radar_45'))
        self.radar_90.listen(lambda radar_data: self.radar_data(radar_data, key='radar_90'))
        self.radar_135.listen(lambda radar_data: self.radar_data(radar_data, key='radar_135'))
        self.radar_180.listen(lambda radar_data: self.radar_data(radar_data, key='radar_180'))
        self.radar_225.listen(lambda radar_data: self.radar_data(radar_data, key='radar_225'))
        self.radar_270.listen(lambda radar_data: self.radar_data(radar_data, key='radar_270'))
        self.radar_315.listen(lambda radar_data: self.radar_data(radar_data, key='radar_315'))

        time.sleep(_SLEEP_TIME_)

        # -------------------------- GETTING CURRENT STATE ON START OF NEW EPISODE ------------------------------

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        current_state, current_state_dict = self.get_current_state()

        self.last_distance_to_goal = current_state_dict['distance_to_goal'] 
        self.distance_to_goal = self.last_distance_to_goal
        self.non_moving_steps_cnt = 0

        return current_state

    def collision_data(self, collision_data):

        """
        Function for storing new collision event data. Intensity of collision normal 
        impulse is stored.
            
        :params:
            - collision_data: carla.CollisionEvent object with information of collision 

        :return:
            None

        """

        imp_3d = collision_data.normal_impulse
        intesity = np.sqrt((imp_3d.x)**2 + (imp_3d.y)**2 + (imp_3d.z)**2)

        self.last_collision_impulse = self.collision_impulse

        self.collision_impulse = intesity

    def radar_data(self, radar_data, key):

        """
        Function for processing and storing radar readings.
            
        :params:
            - radar_data: carla.RadarMeasurement object --> array of carla.RadarDetection objects
                          containg readings from one radar sensor
            - key: key value for self.radar_readings dictionary 

        :return:
            None

        """

        radar_points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')).reshape((len(radar_data), 4))

        if radar_points.shape[0] > 0:
            min_depth_radar_reading = min(np.reshape(radar_points[:,3],(len(radar_data),)))
        else:
            min_depth_radar_reading = 100.0

        self.radar_readings[key] = min_depth_radar_reading

    def transform_angle(self, angle):

        """
        Function for shifting angles.
            
        :params:
            - angle: float angle from range -180 deg to 180 deg

        :return:
            - angle_360: angle shifted to the range 0 deg to 360 deg

        """

        angle_360 = 360 + angle if angle < 0 else angle
        return angle_360

    def get_current_state(self):

        """
        Function for getting current state of Carla vehicle (agent).
            
        :params:
            None

        :return:
            - current_state: numpy array with shape (STATE_SIZE, ) containing all sensor readings:
                             - 8 radar readings:  min depth of radar readings from 8 different positions on vehicle
                             - x: x coordinate of center of vehicle in global coordinates
                             - y: y coordinate of center of vehicle in global coordinates
                             - x_rel: difference of x coordinate center of goal parking spot and of center of vehicle in global coordinates (relative x)
                             - y_rel: difference of y coordinate center of goal parking spot and of center of vehicle in global coordinates (relative y)
                             - angle: angle of rotation along axial axis (z-axis)
                             - vx: linear velocity along x-axis of vehicle
                             - wz: angular velocity along z-axis of vehicle - rotation velocity
                             - distance_to_goal: Euclidian distance from current position of vehicle to the goal position 

            - current_state_dict: dictionary for current state, with same values as current_state numpy array, but with keys

        """

        # -------------------------- GETTING SENSOR READINGS ------------------------------

        current_vehicle_transform = self.vehicle.get_transform()
        current_vehicle_location = current_vehicle_transform.location
        current_vehicle_x = current_vehicle_location.x
        current_vehicle_y = current_vehicle_location.y
        angle = current_vehicle_transform.rotation.yaw

        current_vehicle_linear_velocity = self.vehicle.get_velocity().x
        current_vehicle_angular_velocity = self.vehicle.get_angular_velocity().z

        x = current_vehicle_x
        y = current_vehicle_y
        x_rel = self.parking_map['goal_parking_spot'].location.x - current_vehicle_x
        y_rel = self.parking_map['goal_parking_spot'].location.y - current_vehicle_y
        angle = self.transform_angle(angle)
        vx = current_vehicle_linear_velocity 
        wz = current_vehicle_angular_velocity  
        distance_to_goal = current_vehicle_location.distance(self.parking_map['goal_parking_spot'].location)

        current_state = list(self.radar_readings.values()) + [x, y, x_rel, y_rel, angle, vx, wz, distance_to_goal]

        # -------------------------- PACKING CURRENT STATE IN DICTIONARY AND ARRAY ------------------------------
        sensor_values_dict = {
                                'x': x,
                                'y': y,
                                'x_rel': x_rel,
                                'y_rel': y_rel,
                                'angle': angle,
                                'vx': vx,
                                'wz': wz,
                                'distance_to_goal': distance_to_goal,
                             }

        current_state_dict = self.radar_readings.copy()
        current_state_dict.update(sensor_values_dict)

        current_state = np.array(current_state, dtype='float32').reshape((STATE_SIZE,))

        return current_state, current_state_dict

    def check_non_movement(self):

        """
        Function for logging of how many consecutive steps has agent not moved and indicating if it time to break the episode.
            
        :params:
            None

        :return:
            - non_movement_indicator: boolean value that indicates if agent has violated NON_MOVING_STEPS ban

        """

        non_movement_indicator = False

        if abs(self.last_distance_to_goal - self.distance_to_goal) <= 0.01:

            self.non_moving_steps_cnt += 1

            if self.non_moving_steps_cnt >= NON_MOVING_STEPS:

                non_movement_indicator = True
        else:
            self.non_moving_steps_cnt = 0

        return non_movement_indicator

    def step(self, actions):

        """
        Function for taking provided actions and  collecting penalty/reward for taken actions.
            
        :params:
            - actions: dictionary with 2 elements with keys
                       - 'throttle': throttle value for vehicle from range -1 to 1, negative throttle sets reverse to True
                       - 'steer': steer value for vehicle from range -1 to 1

        :return:
            - current_state: numpy array with shape (STATE_SIZE, ) containing new current state after applied actions
            - reward: reward value for taken actions
            - done: boolean value, indicating if current episode is finished because of bad behavior of agent, or not

        """

        # -------------------------- APPLYING PROVIDED ACTIONS ------------------------------

        reverse = False if actions['throttle'] >= 0 else True
        throttle = abs(actions['throttle'])
        steer = actions['steer']

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, reverse=reverse))

        time.sleep(_SLEEP_TIME_)

        # ---------------- GETTING CURRENT STATE AFTER APPLIED ACTIONS -------------------

        current_state, current_state_dict = self.get_current_state()

        distance = current_state_dict['distance_to_goal']
        angle = current_state_dict['angle']

        self.last_distance_to_goal = self.distance_to_goal
        self.distance_to_goal = distance

        # ---------------- PENALTY/REWARD CALCULATION FOR APPLIED ACTIONS -------------------

        if self.collision_impulse != None and self.collision_impulse != self.last_collision_impulse: 

            if self.collision_impulse >= MAX_COLLISION_IMPULSE:
                reward = -MAX_REWARD
                done = True
            else:
                reward = - 1.0/self.collision_impulse

        elif distance >= MAX_DISTANCE:
            reward = -MAX_REWARD
            done = True 

        elif self.check_non_movement():
            reward = -MAX_REWARD
            done = True 

        else:
            reward = self.calculate_reward(distance, angle)
            done = False

        return current_state, reward, done

    def calculate_reward(self, distance, angle, d_val_1=2, mode='gauss'):

        """
        Function for regular calculating current reward for just taken actions. Check for provided
        reward functions analysis in reward_construcion folder.
            
        :params:
            - distance: Euclidean distance from current agent's position to the center of goal parking spot
            - angle: current agent's global yaw angle
            - d_val_1: distance (in meters) where reward function is crossing 1 (for 'lin' mode)
            - mode: currently 2 modes:
                    - lin: part-by-part linear reward function, with values in range [0,1] for distance in range [MAX_DISTANCE, d_val_1],
                           and values higher than 1 or equal to 1 for distance in range [0, d_val_1]
                    - gauss: Gaussian-like reward function with hyperparameter SIGMA, centered over mean value which is 0.0, because of
                             maximum of this function in distance = 0

        :return:
            - reward: calculated reward value for taken actions

        """

        # ----------------------- ANGLE PENALTY CALCULATION ----------------------------
        theta = self.transform_angle(self.parking_map['goal_parking_spot'].rotation.yaw) - angle
        angle_penalty = abs(np.cos(np.deg2rad(theta)))

        # ----------------------- DISTANCE PENALTY CALCULATION ----------------------------

        if mode == 'lin':
            if distance >= d_val_1 and distance < MAX_DISTANCE :
                reward = (-1.0/(MAX_DISTANCE-d_val_1))*distance + MAX_DISTANCE/(MAX_DISTANCE-d_val_1)
            
            elif distance >= 0 and distance < d_val_1:
                reward = (1-MAX_REWARD)*distance/float(d_val_1) + MAX_REWARD

        elif mode == 'gauss':
            reward = MAX_REWARD*np.exp(-distance**2/(2*SIGMA**2))        

        
        reward = reward * angle_penalty

        return reward

    def destroy_actors(self):

        """
        Function for destroying all Carla actors.
            
        :params:
            None

        :return:
            None

        """

        if self.actor_list:
            for actor in self.actor_list:
                    success = actor.destroy()

# ---------------------------------------------------------------------------------------------------
# ORNSTEIN-UHLENBECK PROCESS NOISE CLASS
# ---------------------------------------------------------------------------------------------------
class OUActionNoise:

    """
    For implementing better exploration by the Actor network it is nice to add some noisy perturabtions.
    This process samples noise from a correlated normal distribution.
    """

    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):

        # https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):

        """
        Function for reseting history of this process.
            
        :params:
            None

        :return:
            None

        """
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mu)
        
# ---------------------------------------------------------------------------------------------------
# DEEP DETERMINISTIC POLICY GRADIENT (DDPG) AGENT CLASS
# ---------------------------------------------------------------------------------------------------
class DDPGAgent:

    """
    Class serves for DDPG reinforcement learning problem solving.
    It uses two dependent neural networks - Actor & Critic and gets trained using DDPG algorithm, 
    target models for both of these 2 neural networks and experience replay memory.
    """

    def __init__(self):
        pass

    def get_actor(self, model_name='', terminated = False, show_summary = False):

        """
        Function for getting actor model.
            
        :params:
            - model_name: string (either '', either '_target'), indicating if it is about classic or target model 
            - terminated: boolean value, indicating if we have to load weights from models stored as terminated 
            - show_summary: boolean value for printing mode summary

        :return:
            - model: model/neural network constructed or loaded from previously stored model

        """

        global training_indicator

        stopped = '_terminated' if terminated else ''
        model_weights_file_name = 'models/parking_agent'+ model_name + '_actor'  + stopped + '.h5'

        # ----------------------- CONSTRUCTING ACTOR MODEL ----------------------------

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(STATE_SIZE))
        normalized_inputs = layers.BatchNormalization()(inputs)
        out1 = layers.Dense(256, activation='relu')(normalized_inputs)
        out2 = layers.Dense(256, activation='relu')(out1)
        # throttle_action = layers.Dense(1, activation='sigmoid', kernel_initializer=last_init)(out2)
        # steer_action = layers.Dense(1, activation='tanh', kernel_initializer=last_init)(out2)
        # outputs = layers.Concatenate()([throttle_action, steer_action])

        outputs = layers.Dense(2, activation='tanh', kernel_initializer=last_init)(out2)

        model = tf.keras.Model(inputs, outputs, name = 'Target_Actor_Model' if model_name=='_target' else 'Actor_Model')

        # ----------------------- LOADING STORED WEIGHTS IF ENABLED ----------------------------

        if training_indicator == 1 and os.path.exists(model_weights_file_name):
            model.load_weights(model_weights_file_name)

        elif model_name == '_target':
            model.set_weights(actor_model.get_weights())

        if show_summary:
            print(model.summary())

        return model

    def get_critic(self, model_name='', terminated = False, show_summary = False):

        """
        Function for getting critic model.
            
        :params:
            - model_name: string (either '', either '_target'), indicating if it is about classic or target model 
            - terminated: boolean value, indicating if we have to load weights from models stored as terminated 
            - show_summary: boolean value for printing mode summary

        :return:
            - model: model/neural network constructed or loaded from previously stored model

        """

        global training_indicator

        stopped = '_terminated' if terminated else ''
        model_weights_file_name = 'models/parking_agent'+ model_name +'_critic' + stopped + '.h5'

        # ----------------------- CONSTRUCTING CRITIC MODEL ----------------------------

        state_input = layers.Input(shape=(STATE_SIZE))
        normalized_state_inputs = layers.BatchNormalization()(state_input)
        state_out1 = layers.Dense(64, activation='relu')(normalized_state_inputs)
        state_out2 = layers.Dense(128, activation='relu')(state_out1)

        action_input = layers.Input(shape=(ACTIONS_SIZE))
        normalized_action_inputs = layers.BatchNormalization()(action_input)
        action_out1 = layers.Dense(128, activation='relu')(normalized_action_inputs)

        concat = layers.Concatenate()([state_out2, action_out1])
        out1 = layers.Dense(512, activation='relu')(concat)
        out2 = layers.Dense(512, activation='relu')(out1)
        outputs = layers.Dense(1)(out2)
        model = tf.keras.Model([state_input, action_input], outputs, name = 'Target_Critic_Model' if model_name=='_target' else 'Critic_Model')
       
        # ----------------------- LOADING STORED WEIGHTS IF ENABLED ----------------------------

        if training_indicator == 1 and os.path.exists(model_weights_file_name):
            model.load_weights(model_weights_file_name)

        elif model_name == '_target':
            model.set_weights(critic_model.get_weights())

        if show_summary:
            print(model.summary())

        return model

    def policy(self, state, noise_objects_dict):

        """
        Function for getting policy (best actions for provided current state).
            
        :params:
            - state: numpy array with shape (STATE_SIZE, ), refering to current state of agent
            - noise_objects_dict: dictionary with two objects of class OUActionNoise with keys
                                  - 'throttle': refering to the noise object for throttle action
                                  - 'steer': refering to the noise object for steer action

        :return:
            - legal_actions_array: numpy array with shape (ACTIONS_SIZE, ), containing sampled actions with added noise
            - legal_actions_dict: dictionary containing elements with values same as legal_actions_array

        """

        global epsilon, training_indicator

        # ----------------------- SAMPLING ORNSTEIN-UHLENBECK NOISE ----------------------------

        noise_throttle = noise_objects_dict['throttle']
        noise_steer= noise_objects_dict['steer']

        # ------------ SAMPLING ACTIONS FROM ACTOR MODEL AND ADDING NOISE TO THEM --------------

        sampled_actions = tf.squeeze(actor_model(state))
        sampled_actions = sampled_actions.numpy()

        epsilon -= 1.0/EXPLORE 

        throttle = float(sampled_actions[0] + float(np.sign(training_indicator)*(training_indicator-1))*max(epsilon, MIN_EPSILON)*noise_throttle())
        steer = float(sampled_actions[1] + float(np.sign(training_indicator)*(training_indicator-1))*max(epsilon, MIN_EPSILON)*noise_steer())

        if throttle > 1:
            throttle = 1
        elif throttle < -1:
            throttle = -1

        if steer > 1:
            steer = 1
        elif steer < -1:
            steer = -1

        # ----------------------- PACKING SAMPLED ACTIONS ----------------------------

        legal_actions_array = np.array([throttle, steer], dtype='float32').reshape((ACTIONS_SIZE,))
        legal_actions_dict = {
                              'throttle': throttle,
                              'steer': steer
                             }

        return legal_actions_array, legal_actions_dict

# ---------------------------------------------------------------------------------------------------
# REPLAY BUFFER CLASS
# ---------------------------------------------------------------------------------------------------
class ReplayBuffer:

    def __init__(self, buffer_capacity=100000, batch_size=64):

        """
        Constructor of ReplayBuffer class.
            
        :params:
            - buffer_capacity: capacity of memory for learning process
            - batch_size: size of batch sampling from memory for learning over them

        :return:
            None

        """

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))
        self.action_buffer = np.zeros((self.buffer_capacity, ACTIONS_SIZE))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))

    def record(self, observation):

        """
        Function for recording experience.
            
        :params:
            - observation: dictionary with keys:
                           - 'state': numpy array with shape (STATE_SIZE, ), containing current state, for whom best actions are calculated
                           - 'action': numpy array with shape (ACTIONS_SIZE, ), containing sampled actions
                           - 'reward': reward value for sampled actions
                           - 'next_state': numpy array with shape (STATE_SIZE, ), containing new/next state where we came, because of taken actions

        :return:
            None

        """

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = observation['state']
        self.action_buffer[index] = observation['action']
        self.reward_buffer[index] = observation['reward']
        self.next_state_buffer[index] = observation['next_state']

        self.buffer_counter += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):

        """
        Function for training and updating  Actor and Critic models.
        Function with TensorFlow Eager execution.
            
        :params:
            - state_batch: batch of current states dragged from replay buffer
            - action_batch: batch of actions dragged from replay buffer
            - reward_batch: batch of rewards dragged from replay buffer
            - next_state_batch: batch of next states dragged from replay buffer

        :return:
            None

        """

        # -------------------- TRAINING AND UPDATING CRITIC MODEL ------------------------

        """
        Critic loss - mean squared error of 'y - Q(s, a)' where 'y' is the expected return as 
                      seen by the Target network and 'Q(s, a)' is action value predicted by the Critic network

        'y' is a moving target that the Critic model tries to achieve
        This target is stable because of slowly updating of the Target model 

        """

        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + GAMMA * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        # -------------------- TRAINING AND UPDATING ACTOR MODEL ------------------------

        """
        Actor loss - it's computed using the mean of the value given by the Critic network for the actions taken by the Actor network
                     we want to maximize this number (hence we are using '-value' as we want to maximize the value given by the critic for our actions)

        Updating the Actor network so that it produces actions that get the maximum predicted value as seen by the Critic, for a given state.

        """

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)

            
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    def learn(self):

        """
        Function for learning process. Samping batches from ReplayBuffer, converting
        them to tensors, calculate loss and updating parameters of networks.
            
        :params:
            None

        :return:
            None

        """

        # ---------- GETTING DATA FROM MEMORY AND CONVERTING THEM INTO TENSORS ----------
        
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])


        # ---------- LOSS CALCUATION AND UPDATING PARAMETERS ----------

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

@tf.function
def update_target(target_weights, actual_weights):

    """
    Function for slowly updating weights on target models depending on actual models and parameter TAU.
    This is very useful for stability of algorithm (developed by DeepMind, as an upgrade of Q-learning)
    Function with TensorFlow Eager execution.
        
    :params:
        - target_weights: weights of target model
        - actual_weights: weights of actual model

    :return:
        None

    """

    for (a, b) in zip(target_weights, actual_weights):
        a.assign(b * TAU + a * (1 - TAU))

# ---------------------------------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    random.seed(3)
    np.random.seed(3)
    tf.random.set_seed(3)

    # ----------------------- GPU ACCELERATION SETTINGS ----------------------------
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = MEMORY_FRACTION
    session = tf.compat.v1.InteractiveSession(config=config)

    # ----------------------- CREATING NECESSARY FOLDERS ----------------------------
    if not os.path.isdir('models'):
        os.makedirs('models')

    # creating training_images folder
    if not os.path.isdir('training_images'):
        os.makedirs('training_images')


    # ---------------------- SELECTING PURPOSE OF PROGRAM ----------------------

    training_indicator = int(input('Select one option: \n \tPlay with trained model (press 0)\n \tTrain pretrained model (press 1)\n \tTrain new model (press 2)\n \tYour answer: '))

    try: 

        # ------------ CREATING ENVIRONMENT, AGENT AND NOISE OBJECTS ----------------
        env = CarlaEnvironment()
        agent = DDPGAgent()

        ou_noise_throttle = OUActionNoise(mu=np.zeros(1), sigma=0.4*np.ones(1))
        ou_noise_steer = OUActionNoise(mu=np.zeros(1), sigma=0.3*np.ones(1))

        ou_noise_dict = {
                          'throttle': ou_noise_throttle,
                          'steer': ou_noise_steer
                         }

        # -------------- CREATING/LOADING ALL MODELS, CREATING REPLAY MEMORY ------------------

        actor_model = agent.get_actor()
        critic_model = agent.get_critic()

        target_actor = agent.get_actor(model_name='_target')
        target_critic = agent.get_critic(model_name='_target')

        actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)
        critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)

        replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY, BATCH_SIZE)

        # ----------------- TRAINING PROCESS (ITERATING OVER EPISODES) ---------------------

        episode_reward_list = []
        average_reward_list = []

        if training_indicator in [1, 2]:
            print('-----------------Start of training process---------------')
        else:
            print('-----------------Start---------------')

        for episode in range(1,TOTAL_EPISODES+1):

            state = env.reset()
            episodic_reward = 0

            # ----------------- ITERATING IN ONE EPISODE ---------------------
            for step in range(STEPS_PER_EPISODE):

                # ----------------- SAMPLING AND APPLYING ACTIONS, TAKING OBSERVATIONS ---------------------

                tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

                actions_arr, actions_dict = agent.policy(tf_state, ou_noise_dict)
                next_state, reward, done = env.step(actions_dict)

                observation = {
                                'state': state,
                                'action': actions_arr,
                                'reward': reward,
                                'next_state': next_state
                              }

                # ----------------- RECORDING CURRENT OBSERVATION ---------------------

                replay_buffer.record(observation)

                episodic_reward += reward

                # ----------------------- LEARNING PROCESS ----------------------------
                if training_indicator in [1, 2]:
                    
                    replay_buffer.learn()

                    # -------------------- UPDATING TARGET MODELS -------------------------
                    update_target(target_actor.variables, actor_model.variables)
                    update_target(target_critic.variables, critic_model.variables)


                print('Current step: %d/%d' %(step, STEPS_PER_EPISODE), end='\r')

                # ------------ EPISODE TERMINATION / TRANSITION -------------
                if done:
                    break

                state = next_state

            env.destroy_actors()

            # ----------------------- STORING REWARDS ----------------------------

            episode_reward_list.append(episodic_reward)
            average_reward = np.mean(episode_reward_list[-AVERAGE_EPISODES_COUNT:])
            average_reward_list.append(average_reward)

            print('Episode * {} * Episodic Reward is ==> {}'.format(episode, episodic_reward))            

        if training_indicator in [1, 2]:
            print('-----------------End of training process---------------')
        else:
            print('-----------------End---------------')

        # ----------------------- SAVING MODELS ----------------------------

        actor_model.save_weights('models/parking_agent_actor.h5')
        critic_model.save_weights('models/parking_agent_critic.h5')

        target_actor.save_weights('models/parking_agent_target_actor.h5')
        target_critic.save_weights('models/parking_agent_target_critic.h5')

        now = datetime.now()
        date_time_string = now.strftime('%d-%m-%Y_%H-%M-%S')

        # ----------------------- PLOTTING REWARDS ----------------------------

        plt.figure(figsize = (10,10), dpi = 100)
        plt.plot(np.arange(1,TOTAL_EPISODES+1), episode_reward_list, color='tomato', linewidth=1.0, label='episodic')
        plt.plot(np.arange(1,TOTAL_EPISODES+1), average_reward_list, color='darkgreen', linewidth=1.5, label='average episodic')
        plt.xlabel('Episode')
        plt.ylabel('Rt')
        plt.grid()
        plt.title('Episodic and Average Episodic Reward \n (average over every last {} episodes)'.format(AVERAGE_EPISODES_COUNT))
        plt.legend(loc='upper right')
        plt.savefig(FOLDER_PATH + '/training_images/training_rewards_'+ date_time_string +'.png')

    # ----------------------- CATCHING EXCEPTIONS DURING TRAINING ----------------------------
    except :
        
        if training_indicator == 1:

            # ----------------------- SAVING TERMINATED MODELS IN OLD ONES ----------------------------.

            actor_model.save_weights('models/parking_agent_actor.h5')
            critic_model.save_weights('models/parking_agent_critic.h5')

            target_actor.save_weights('models/parking_agent_target_actor.h5')
            target_critic.save_weights('models/parking_agent_target_critic.h5')

        if training_indicator == 2:

            # ----------------------- SAVING TERMINATED MODELS ----------------------------.

            actor_model.save_weights('models/parking_agent_actor_terminated.h5')
            critic_model.save_weights('models/parking_agent_critic_terminated.h5')

            target_actor.save_weights('models/parking_agent_target_actor_terminated.h5')
            target_critic.save_weights('models/parking_agent_target_critic_terminated.h5')

        else:
            pass

        # ----------------------- PLOTTING REWARDS OF TERMINATED LEARNING PROCESS ----------------------------

        now = datetime.now()
        date_time_string = now.strftime('%d-%m-%Y_%H-%M-%S')

        plt.figure(figsize = (10,10), dpi = 100)
        plt.plot(np.arange(1, len(episode_reward_list)+1), episode_reward_list, color='tomato', linewidth=1.0, label='episodic')
        plt.plot(np.arange(1, len(average_reward_list)+1), average_reward_list, color='darkgreen', linewidth=1.5, label='average episodic')
        plt.xlabel('Episode')
        plt.ylabel('Rt')
        plt.grid()
        plt.title('Episodic and Average Episodic Reward \n (averaged over every last {} episodes) \n --- terminated at episode {}/{} ---'.format(AVERAGE_EPISODES_COUNT, episode-1, TOTAL_EPISODES))
        plt.legend(loc='upper right')
        plt.savefig(FOLDER_PATH + '/training_images/training_rewards_terminated_'+ date_time_string +'.png')

