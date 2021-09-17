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

TRAINING_INDICATOR = 2
SELECTED_MODEL = 'only_throttle'
TRAINING_NAME = 'training'
SELECTED_SPAWNING_METHOD = 0

ACTIONS_SIZE = 1
STATE_SIZE = 16

MAX_COLLISION_IMPULSE = 50
MAX_DISTANCE = 20
MAX_REWARD = 20.0
SIGMA = 2.0

# ---------------------------------------------------------------------------------------------------
# GLOBAL VARIABLES FOR TRAINING 
# ---------------------------------------------------------------------------------------------------
MEMORY_FRACTION = 0.3333

TOTAL_EPISODES = 1000 
STEPS_PER_EPISODE = 100
AVERAGE_EPISODES_COUNT = 40

CORRECT_POSITION_NON_MOVING_STEPS = 5
OFF_POSITION_NON_MOVING_STEPS = 20

REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 64

CRITIC_LR = 0.002
ACTOR_LR = 0.001
GAMMA = 0.99
TAU = 0.005

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
            - parking_map: dictionary with 4 specific carla.Location objects for corners of goal parking spot
                           and one carla.Transform object for center of goal parking spotv

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
                       'goal_parking_spot'  : goal_parking_spot
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
                                                    'x_min': 0,
                                                    'x_max': 16,
                                                    'y_min': -36,
                                                    'y_max': -28,
                                                    'yaw_min': [150,-30],
                                                    'yaw_max': [210,30]
                                                  }
                            }

        spawn_transform = self.get_spawn_transform(mode_values_dict[mode], mode)

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

    def get_spawn_transform(self, values, mode):

        """
        Function for generating random spawn transform for vehicle
        depending of input values.
            
        :params:
            - values: dictionary with coordintes of corners for spawning and initial offsets
            - mode: 3 modes are currently provided:
                    - carla_recommended: spawn points near parking recommended by Carla authors
                    - random_lane: spawn points in lane closest to the parking
                    - random_entrance: spawn points in spatial rectangle in the entrance

        :return:
            - spawn_transform: carla.Transform object for spawning location and rotation

        """

        x0 = values['x0']
        y0 = values['y0']

        x_min = values['x_min']
        x_max = values['x_max']

        y_min = values['y_min']
        y_max = values['y_max']



        if mode == 'random_entrance':

            index = random.choice([0, 1])

            yaw_min = values['yaw_min'][index]
            yaw_max = values['yaw_max'][index]

        else:
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

    def reset(self, spawn_point = None):

        """
        Function for reseting environment and starting new episode.
            
        :params:
            - spawn_point: specific spawn point to spawn vehicle on start of episode (carla.Transform object)

        :return:
            - current_state: numpy array with shape (STATE_SIZE, ) 
                             with all sensor readings on start of the new episode
            - spawn_point: finally used spawn point for spawning vehicle (carla.Transform object) on episode start

        """

        self.collision_impulse = None
        self.last_collision_impulse = None

        self.actor_list = []
 
        # ------------------------------ SPAWNING AGENT ----------------------------------

        if spawn_point == None:

            if SELECTED_SPAWNING_METHOD == 1 :
                spawn_point = self.random_spawn('random_entrance')

            else:
                if SELECTED_MODEL == 'only_throttle':
                    spawn_point = carla.Transform(carla.Location(x=17.2, y=-29.7, z=self.spawning_z_offset), carla.Rotation(yaw=random.choice([0.0, 180.0])))
                else: 
                    spawn_point = carla.Transform(carla.Location(x=17.2, y=-29.7, z=self.spawning_z_offset), carla.Rotation(yaw=180.0))

        self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
        self.actor_list.append(self.vehicle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(_SLEEP_TIME_)

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

        self.last_distance_to_goal = None
        self.distance_to_goal = current_state_dict['distance_to_goal'] 
        self.initial_distance = self.distance_to_goal
        
        self.last_angle = None
        self.angle = current_state_dict['angle'] 

        self.non_moving_steps_cnt = 0

        return current_state, spawn_point

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
            - correct_position_non_movement_indicator: boolean value that indicates if agent has not had moved for CORRECT_POSITION_NON_MOVING_STEPS steps in correct position
            - off_position_non_movement_indicator: boolean value that indicates if agent has not had moved for OFF_POSITION_NON_MOVING_STEPS steps in any other position but correct

        """

        correct_position_non_movement_indicator = False
        off_position_non_movement_indicator = False

        if abs(self.last_distance_to_goal - self.distance_to_goal) <= 0.05:

            self.non_moving_steps_cnt += 1

            if self.check_if_parked() and (self.non_moving_steps_cnt >= CORRECT_POSITION_NON_MOVING_STEPS):

                correct_position_non_movement_indicator = True

            elif (not self.check_if_parked()) and (self.non_moving_steps_cnt >= OFF_POSITION_NON_MOVING_STEPS):

                off_position_non_movement_indicator = True
        else:
            self.non_moving_steps_cnt = 0

        return correct_position_non_movement_indicator, off_position_non_movement_indicator

    def check_if_parked(self):

        """
        Function for checking if agent has parked or come on goal parking spot.
            
        :params:
            None

        :return:
            - vehicle_parked: boolean value indicating if agent has parked pretty well on goal parking spot

        """

        vehicle_parked = False

        goal_angle = self.transform_angle(self.parking_map['goal_parking_spot'].rotation.yaw)

        if (self.distance_to_goal <= 0.5) and ((abs(goal_angle - self.angle) <= 20) or (abs(goal_angle - self.angle) >= 160)) :

            vehicle_parked = True

        return vehicle_parked

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

        angle_penalty = max(angle_penalty, 0.01)

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

    def apply_vehicle_actions(self, throttle=0.0, steer=0.0, reverse=False, brake=0.0, sleep_time=None):

        """
        Function for taking actions of Carla agent, proposed by Actor model, or recoreded.
        This function takes care of vehicle control while training or while recording, but not
        while reseting environment.
            
        :params:
            - throttle: value for throttle action
            - steer: value for steer action
            - reverse: boolean value for reverse indicator
            - brake: value for brake action
            - sleep_time: duration of this particular action

        :return:
            None

        """

        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake), reverse=reverse))

        if sleep_time == None:
            time.sleep(_SLEEP_TIME_)
        else:
            time.sleep(sleep_time)

    def step(self, actions, current_step):

        """
        Function for taking provided actions and  collecting penalty/reward for taken actions.
            
        :params:
            - actions: dictionary with 2 elements with keys
                       - 'throttle': throttle value for vehicle from range -1 to 1, negative throttle sets reverse to True
                       - 'steer': steer value for vehicle from range -1 to 1
            - current_step: value of current step over one episode

        :return:
            - current_state: numpy array with shape (STATE_SIZE, ) containing new current state after applied actions
            - reward: reward value for taken actions
            - done: boolean value, indicating if current episode is finished because of bad behavior of agent, or not
            - info: information of this step taken
            - record_episode: integer from set {-1, 0, 1}: -1 - agent is not parked the best, but if episodic reward if great, then record
                                                            0 - agent has not moved from start of episode, because it is near goal, do not record
                                                            1 - agent is parked well, record it


        """

        # -------------------------- APPLYING PROVIDED ACTIONS ------------------------------

        reverse = False if actions['throttle'] >= 0 else True
        throttle = abs(actions['throttle'])
        steer = actions['steer']

        self.apply_vehicle_actions(throttle, steer, reverse)

        # ---------------- GETTING CURRENT STATE AFTER APPLIED ACTIONS -------------------

        current_state, current_state_dict = self.get_current_state()

        distance = current_state_dict['distance_to_goal']
        angle = current_state_dict['angle']

        self.last_distance_to_goal = self.distance_to_goal
        self.distance_to_goal = distance

        self.last_angle = self.angle
        self.angle = angle

        done = False
        record_episode = -1

        correct_position_stagnating, off_position_stagnating = self.check_non_movement()

        # ---------------- PENALTY/REWARD CALCULATION FOR APPLIED ACTIONS -------------------

        if ((self.collision_impulse != None) and (self.collision_impulse != self.last_collision_impulse) and (distance < MAX_DISTANCE)): 

            if self.collision_impulse >= MAX_COLLISION_IMPULSE:
                reward = -1000
                done = True
                info = 'Agent crashed :('

            else:
                reward = -self.collision_impulse/MAX_COLLISION_IMPULSE
                done = False
                info = 'Agent crahed slightly :/'

        elif distance >= MAX_DISTANCE:
            reward = -500
            done = True    
            info = 'Agent moved far away from goal :(' 

        elif correct_position_stagnating == True:
            reward = 1000
            done = True 
            info = 'Great job! Agent parked! :D' 

            if abs(self.initial_distance - distance) > 0.1:
                record_episode = 1
            else:
                record_episode = 0

        elif off_position_stagnating == True:
            reward = 0
            done = True 
            info = 'Agent stayed for to long off goal position (distance = ' + str(distance) +'):/' 

            if abs(self.initial_distance - distance) > 0.1:
                record_episode = -1
            else:
                record_episode = 0

        else:
            reward = self.calculate_reward(distance, angle, mode='lin')
            done = False
            info = 'Agent still learning :)' 

        # ------------------- ADDITIONAL NEGATIVE REWARDS FOR ONLY THROTTLE MODEL ---------------------

        if SELECTED_MODEL == 'only_throttle' and SELECTED_SPAWNING_METHOD == 0: 

            if current_state_dict['angle'] in [0.0, 360.0]  : # agent should go back for goal

                if (reverse == False) and (distance > 2) and (current_state_dict['x_rel'] < 0):
                        if reward > 0 :
                            reward *= -1.0
                elif (reverse == True) and (distance > 2) and (current_state_dict['x_rel'] > 0):
                        if reward > 0 :
                            reward *= -1.0

            else: # agent should go back for goal

                if (reverse == True) and (distance > 2) and (current_state_dict['x_rel'] < 0):
                    reward = -1.0*reward if reward > 0 else reward

                elif (reverse == False) and (distance > 2) and (current_state_dict['x_rel'] > 0):
                    reward = -1.0*reward if reward > 0 else reward

        return current_state, reward, done, info, record_episode

    def play_recording(self, recording):

        """
        Function for playing recorded set of actions.
            
        :params:
            - recording: list of consecutive actions recoreded while training

        :return:
            None

        """

        for action in recording:
            reverse = False if action[0] >= 0 else True
            throttle = abs(action[0])
            steer = action[1]

            self.apply_vehicle_actions(throttle, steer, reverse)

        self.apply_vehicle_actions(brake=1.0, sleep_time=5.0)

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

    def noise_factor(self):

        """
        Function for calculating factor that multiplies noise sample while sampling actions.
            
        :params:
            None

        :return:
            - factor: calculated multiplicative factor

        """

        global epsilon, TRAINING_INDICATOR

        if TRAINING_INDICATOR == 1:
            factor = MIN_EPSILON

        elif TRAINING_INDICATOR == 2:

            epsilon -= 1.0/EXPLORE 
            factor = max(epsilon, MIN_EPSILON)

        return factor
    
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

        global TRAINING_INDICATOR, SELECTED_MODEL

        stopped = '_terminated' if terminated else ''
        model_weights_file_name = 'models/' + SELECTED_MODEL + '/parking_agent'+ model_name + '_actor'  + stopped + '.h5'

        # ----------------------- CONSTRUCTING ACTOR MODEL ----------------------------

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(STATE_SIZE))
        normalized_inputs = layers.BatchNormalization()(inputs)
        out1 = layers.Dense(512, activation='relu')(normalized_inputs)
        out2 = layers.Dense(256, activation='relu')(out1)
        outputs = layers.Dense(ACTIONS_SIZE, activation='tanh', kernel_initializer=last_init)(out2)

        model = tf.keras.Model(inputs, outputs, name = 'Target_Actor_Model' if model_name=='_target' else 'Actor_Model')

        # ----------------------- LOADING STORED WEIGHTS IF ENABLED ----------------------------

        if TRAINING_INDICATOR == 1 and os.path.exists(model_weights_file_name):
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

        global TRAINING_INDICATOR, SELECTED_MODEL

        stopped = '_terminated' if terminated else ''
        model_weights_file_name = 'models/' + SELECTED_MODEL + '/parking_agent'+ model_name +'_critic' + stopped + '.h5'

        # ----------------------- CONSTRUCTING CRITIC MODEL ----------------------------

        state_input = layers.Input(shape=(STATE_SIZE))
        normalized_state_inputs = layers.BatchNormalization()(state_input)
        state_out1 = layers.Dense(64, activation='relu')(normalized_state_inputs)
        state_out2 = layers.Dense(128, activation='relu')(state_out1)

        action_input = layers.Input(shape=(ACTIONS_SIZE))
        normalized_action_inputs = layers.BatchNormalization()(action_input)
        action_out1 = layers.Dense(64, activation='relu')(normalized_action_inputs)
        action_out2 = layers.Dense(128, activation='relu')(action_out1)

        concat = layers.Concatenate()([state_out2, action_out2])
        out1 = layers.Dense(256, activation='relu')(concat)
        out2 = layers.Dense(256, activation='relu')(out1)
        outputs = layers.Dense(1)(out2)
        model = tf.keras.Model([state_input, action_input], outputs, name = 'Target_Critic_Model' if model_name=='_target' else 'Critic_Model')
       
        # ----------------------- LOADING STORED WEIGHTS IF ENABLED ----------------------------

        if TRAINING_INDICATOR == 1 and os.path.exists(model_weights_file_name):
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

        global SELECTED_MODEL

        # ----------------------- SAMPLING ORNSTEIN-UHLENBECK NOISE ----------------------------

        noise_throttle = noise_objects_dict['throttle']
        noise_steer= noise_objects_dict['steer']

        # ------------ SAMPLING ACTIONS FROM ACTOR MODEL AND ADDING NOISE TO THEM --------------

        sampled_actions = tf.squeeze(actor_model(state))
        sampled_actions = sampled_actions.numpy()

        if SELECTED_MODEL == 'only_throttle':
            throttle = float(sampled_actions + float(noise_throttle.noise_factor()*noise_throttle()))

            if throttle > 1:
                throttle = 1
            elif throttle < -1:
                throttle = -1

            steer = float(0.0)

            legal_actions_array = np.array([throttle], dtype='float32').reshape((ACTIONS_SIZE,))

        elif SELECTED_MODEL == 'throttle_and_steer':

            throttle = float(sampled_actions[0] + float(noise_throttle.noise_factor()*noise_throttle()))
            steer = float(sampled_actions[1] + float(noise_steer.noise_factor()*noise_steer()))

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

    def save_recording(self, actions_list, episode, spawn_point):

        """
        Function for saving all actions agent had taken for getting to the goal.
        Episode is recorded only if agent manages to solve problem, to get correctly to the pakring spot.
        If agent make some small crashes but still manages to find the way to goal, episode will still be recorded.

        :params:
            - actions_list: list of actions taken to get to the goal
            - episode: ordinal number of episode 
            - spawn_point: spawn point of episode (carla.Transform)

        :return:
            None

        """

        global SELECTED_MODEL, SELECTED_SPAWNING_METHOD

        spawn = 'in_front_goal_spawn' if SELECTED_SPAWNING_METHOD == 0 else 'random_spawn'

        actions = np.array(actions_list).reshape((len(actions_list), 2))
        np.savetxt(FOLDER_PATH + '/recordings/' + SELECTED_MODEL + '/' + spawn + '/' + TRAINING_NAME + '/' +  str(episode) + '.csv', actions, delimiter=',')

        d = {
             'x'  : [spawn_point.location.x],
             'y'  : [spawn_point.location.y],
             'z'  : [spawn_point.location.z],
             'yaw': [spawn_point.rotation.yaw]
             }

        df = pd.DataFrame(d)
        df.dtype = 'float32'

        df.to_csv(FOLDER_PATH + '/recordings/' + SELECTED_MODEL + '/' + spawn + '/' + TRAINING_NAME + '/' + str(episode) + '_spawn_point.csv', index=False)

    def get_recordings(self):

        """
        Function for getting saved recordings of agent best behavior.

        :params:
            None

        :return:
            - recordings: list of all recordings
            - spawn_points: list of carla.Transforms for all recordings
            - names: list of names of all recordings

        """

        global SELECTED_MODEL, SELECTED_SPAWNING_METHOD, TRAINING_NAME

        spawn = 'in_front_goal_spawn' if SELECTED_SPAWNING_METHOD == 0 else 'random_spawn'
        path = FOLDER_PATH + '/recordings/' + SELECTED_MODEL + '/' + spawn + '/' + TRAINING_NAME + '/'

        recordings = []
        spawn_points = []
        names = []

        try:
            for filename in os.listdir(path):

                if filename.endswith('.csv') and not filename.endswith('_spawn_point.csv'):

                    episode_number = filename[:-4]

                    rec = np.loadtxt(os.path.join(path, filename), delimiter=',')
                    rec = (np.reshape(rec, (rec.shape[0], 2))).tolist()

                    recordings.append(rec)

                    df = pd.read_csv(os.path.join(path, episode_number + '_spawn_point.csv'))

                    spawn_point = carla.Transform(carla.Location(x=float(df['x'][0]), y=float(df['y'][0]), z=float(df['z'][0])), carla.Rotation(yaw=float(df['yaw'][0])))
                    spawn_points.append(spawn_point)
                    names.append(episode_number)
                     
                else:
                    continue

        except Exception as e: 
            print(e)

        return recordings, spawn_points, names

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

        global TRAINING_INDICATOR, SELECTED_MODEL

        self.batch_size = batch_size
        self.buffer_counter = 0

        if TRAINING_INDICATOR == 1:

            try:

                self.state_buffer = np.loadtxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/state_buffer.csv', delimiter=',')

                if self.state_buffer.shape[0] < buffer_capacity:
                    self.buffer_capacity = buffer_capacity
                    self.buffer_counter = state_buffer.shape[0]-1
                else:
                    self.buffer_capacity = self.state_buffer.shape[0]
                    self.buffer_counter = 0

                self.state_buffer = self.state_buffer.reshape((self.buffer_capacity, STATE_SIZE))
                self.action_buffer = np.loadtxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/action_buffer.csv', delimiter=',').reshape((self.buffer_capacity, ACTIONS_SIZE))
                self.reward_buffer = np.loadtxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/reward_buffer.csv', delimiter=',').reshape((self.buffer_capacity, 1))
                self.next_state_buffer = np.loadtxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/next_state_buffer.csv', delimiter=',').reshape((self.buffer_capacity, STATE_SIZE))

            except:

                self.buffer_capacity = buffer_capacity

                self.state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))
                self.action_buffer = np.zeros((self.buffer_capacity, ACTIONS_SIZE))
                self.reward_buffer = np.zeros((self.buffer_capacity, 1))
                self.next_state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))


        else:
            self.buffer_capacity = buffer_capacity

            self.state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))
            self.action_buffer = np.zeros((self.buffer_capacity, ACTIONS_SIZE))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros((self.buffer_capacity, STATE_SIZE))

    def save_buffer(self):

        """
        Function for saving current replay buffer into .csv file.
            
        :params:
            None

        :return:
            None

        """

        global SELECTED_MODEL

        np.savetxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/state_buffer.csv', self.state_buffer, delimiter=',')
        np.savetxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/action_buffer.csv', self.action_buffer, delimiter=',')
        np.savetxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/reward_buffer.csv', self.reward_buffer, delimiter=',')
        np.savetxt(FOLDER_PATH + '/replay_buffer_data/' + SELECTED_MODEL + '/next_state_buffer.csv', self.next_state_buffer, delimiter=',')

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

def save_training_data(data, column_name, file_name):

    """
    Function for saving training data into .csv file.
        
    :params:
        - data: list of floats
        - column_name: name of the column (data name)
        - file_name: file where data is saved

    :return:
        None

    """

    if not os.path.isdir('training_data/data/' + TRAINING_NAME):
        os.makedirs('training_data/data/' + TRAINING_NAME)

    path = FOLDER_PATH + '/training_data/data/' + TRAINING_NAME + '/' + file_name

    d = {column_name: data}

    df = pd.DataFrame(d)
    df.dtype = 'float32'

    df.to_csv(path, index=False)

def process_init_inputs():

    """
    Function for processing inital inputs for obtaining users purpose of program.
        
    :params:
        None

    :return:
        None

    """

    global TRAINING_INDICATOR, SELECTED_MODEL, RANDOM_SPAWN, TRAINING_NAME, ACTIONS_SIZE, TOTAL_EPISODES, SELECTED_SPAWNING_METHOD

    training_indicator = int(input('Select one option: \n \tPlay with trained model (press 0)\n \tTrain pretrained model (press 1)\n \tTrain new model (press 2)\n \tYour answer: '))

    TRAINING_INDICATOR = training_indicator if training_indicator in [0, 1, 2] else 2

    if TRAINING_INDICATOR in [1, 2]:

        total_episodes = input('Enter the number of total episodes (for default [1000] press Enter):\n \tYour answer: ')
        
        if total_episodes and int(total_episodes) > TOTAL_EPISODES:
            TOTAL_EPISODES = int(total_episodes)

        selected_model = int(input('Select model for training: \n \tOnly throttle action model (press 1)\n \tThrottle and steer actions model (press 2)\n \tYour answer: '))

        SELECTED_MODEL = 'throttle_and_steer' if selected_model == 2 else 'only_throttle'

        if SELECTED_MODEL == 'only_throttle':
            ACTIONS_SIZE = 1

        elif SELECTED_MODEL == 'throttle_and_steer':
            ACTIONS_SIZE = 2

            selected_spawning_method = int(input('Select spawning method: \n \tAlways in front of goal (press 1)\n \tRandomly somewhere around goal (press 2)\n \tYour answer: '))

            selected_spawning_method = selected_spawning_method if selected_spawning_method in [1, 2] else 1

            if selected_spawning_method == 1:
                SELECTED_SPAWNING_METHOD = 0

            elif selected_spawning_method == 2:
                SELECTED_SPAWNING_METHOD = 1

        TRAINING_NAME = str(input('Write the name of this training: \n \tYour answer: '))

    else:

        selected_model = int(input('Select model for playing: \n \tOnly throttle action model (press 1)\n \tThrottle and steer actions model (press 2)\n \tYour answer: '))

        SELECTED_MODEL = 'throttle_and_steer' if selected_model == 2 else 'only_throttle'

        if SELECTED_MODEL == 'only_throttle':
            ACTIONS_SIZE = 1

        elif SELECTED_MODEL == 'throttle_and_steer':
            ACTIONS_SIZE = 2

            selected_spawning_method = int(input('Select spawning method: \n \tAlways in front of goal (press 0)\n \tRandomly somewhere around goal (press 1)\n \tYour answer: '))

            SELECTED_SPAWNING_METHOD = selected_spawning_method if selected_spawning_method in [0, 1] else 0

        TRAINING_NAME = str(input('Write the name of training you want to play: \n \tYour answer: '))
 

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
        os.makedirs('models/only_throttle')
        os.makedirs('models/throttle_and_steer')

    if not os.path.isdir('replay_buffer_data'):
        os.makedirs('replay_buffer_data/only_throttle')
        os.makedirs('replay_buffer_data/throttle_and_steer')

    if not os.path.isdir('recordings'):
        os.makedirs('recordings/only_throttle/in_front_goal_spawn')
        os.makedirs('recordings/throttle_and_steer/random_spawn')
        os.makedirs('recordings/throttle_and_steer/in_front_goal_spawn')

    # ------------------- PROCESSING INITIAL INPUTS OF PROGRAM -------------------
    process_init_inputs()

    # ------------ CREATING ENVIRONMENT, AGENT AND NOISE OBJECTS ----------------
    env = CarlaEnvironment()
    agent = DDPGAgent()

    ou_noise_throttle = OUActionNoise(mu=np.zeros(1), sigma=0.4*np.ones(1))
    ou_noise_steer = OUActionNoise(mu=np.zeros(1), sigma=0.3*np.ones(1))

    ou_noise_dict = {
                      'throttle': ou_noise_throttle,
                      'steer': ou_noise_steer
                     }

    # ------------------- PLAYING RECOREDED BEST EPISODES -------------------

    if TRAINING_INDICATOR == 0:

        try:

            print('-----------------Playing started---------------')

            recordings, spawn_points, names = agent.get_recordings()

            if not recordings:
                print('No recordings to play')
            else:

                current_rec = 1
                for recording, spawn_point, name in zip(recordings, spawn_points, names):
                    _, _ = env.reset(spawn_point)
                    print('Current recording: %s ---> %d/%d' %(name, current_rec, len(recordings)), end='\r')
                    env.play_recording(recording)

                    env.destroy_actors()

                    current_rec += 1

                print('-----------------Playing finished---------------')

        except Exception as e:
            print('Failed to play agent recordings!')
            print(e)

    else:

        try:

            # -------------- CREATING/LOADING ALL MODELS, CREATING REPLAY MEMORY ------------------

            actor_model = agent.get_actor(show_summary=True)
            critic_model = agent.get_critic(show_summary=True)

            target_actor = agent.get_actor(model_name='_target')
            target_critic = agent.get_critic(model_name='_target')

            actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)
            critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)

            replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY, BATCH_SIZE)

            # ----------------- TRAINING PROCESS (ITERATING OVER EPISODES) ---------------------

            reward_list = []
            step_list = []
            episode_reward_list = []
            average_reward_list = []

            print('-----------------Start of training process---------------')

            for episode in range(1,TOTAL_EPISODES+1):

                state, spawn_point = env.reset()
                episodic_reward = 0

                actions_list = []

                # ----------------- ITERATING IN ONE EPISODE ---------------------

                for step in range(1,STEPS_PER_EPISODE+1):

                    # ----------------- SAMPLING AND APPLYING ACTIONS, TAKING OBSERVATIONS ---------------------

                    tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

                    actions_arr, actions_dict = agent.policy(tf_state, ou_noise_dict)
                    next_state, reward, done, info, record_episode = env.step(actions_dict, step)

                    observation = {
                                    'state': state,
                                    'action': actions_arr,
                                    'reward': reward,
                                    'next_state': next_state
                                  }

                    actions_list.append(list(actions_dict.values()))

                    # ----------------- RECORDING CURRENT OBSERVATION ---------------------

                    replay_buffer.record(observation)

                    reward_list.append(reward)

                    episodic_reward += reward

                    # ----------------------- LEARNING PROCESS ----------------------------
                        
                    replay_buffer.learn()

                    # -------------------- UPDATING TARGET MODELS -------------------------
                    update_target(target_actor.variables, actor_model.variables)
                    update_target(target_critic.variables, critic_model.variables)
                    
                    # ------------ EPISODE TERMINATION / TRANSITION -------------
                    if done:
                        break
                    else:
                        print('Current step: %d/%d <<<>>> %s' %(step, STEPS_PER_EPISODE, info), end='\r')

                    # -------------- RECORDING EPISODE IF IT IS GOOD ENOUGH -----------------

                    if record_episode == 1 or ((episodic_reward >= 100) and record_episode == -1):
                        agent.save_recording(actions_list, episode, spawn_point)

                    state = next_state

                env.destroy_actors()

                if not step_list:
                    step_list.append(step)
                else:
                    step_list.append(step_list[-1]+step)

                # ----------------------- STORING REWARDS ----------------------------

                episode_reward_list.append(episodic_reward)
                average_reward = np.mean(episode_reward_list[-AVERAGE_EPISODES_COUNT:])
                average_reward_list.append(average_reward)

                print('Episode * {} * Episodic Reward is ==> {} <<<>>> {}'.format(episode, episodic_reward, info))            

            print('-----------------End of training process---------------')

            # ----------------------- SAVING MODELS ----------------------------

            actor_model.save_weights('models/' + SELECTED_MODEL + '/parking_agent_actor.h5')
            critic_model.save_weights('models/' + SELECTED_MODEL + '/parking_agent_critic.h5')

            target_actor.save_weights('models/' + SELECTED_MODEL + '/parking_agent_target_actor.h5')
            target_critic.save_weights('models/' + SELECTED_MODEL + '/parking_agent_target_critic.h5')


            # ----------------------- SAVING TRAINING DATA ----------------------------

            save_training_data(data=reward_list, column_name='rewards', file_name='rewards.csv')
            save_training_data(data=step_list, column_name='step', file_name='steps.csv')
            save_training_data(data=episode_reward_list, column_name='episodic_reward', file_name='episodic_rewards.csv')
            save_training_data(data=average_reward_list, column_name='average_reward', file_name='average_episodic_rewards.csv')

            replay_buffer.save_buffer()

        # ----------------------- CATCHING EXCEPTIONS DURING TRAINING ----------------------------
        except Exception as e:
            
            if TRAINING_INDICATOR == 1:

                # ----------------------- SAVING TERMINATED MODELS IN OLD ONES ----------------------------

                actor_model.save_weights('models/' + SELECTED_MODEL + '/parking_agent_actor.h5')
                critic_model.save_weights('models/' + SELECTED_MODEL + '/parking_agent_critic.h5')

                target_actor.save_weights('models/' + SELECTED_MODEL + '/parking_agent_target_actor.h5')
                target_critic.save_weights('models/' + SELECTED_MODEL + '/parking_agent_target_critic.h5')

            elif TRAINING_INDICATOR == 2:

                # ----------------------- SAVING TERMINATED MODELS ----------------------------.

                actor_model.save_weights('models/' + SELECTED_MODEL + '/parking_agent_actor_terminated.h5')
                critic_model.save_weights('models/' + SELECTED_MODEL + '/parking_agent_critic_terminated.h5')

                target_actor.save_weights('models/' + SELECTED_MODEL + '/parking_agent_target_actor_terminated.h5')
                target_critic.save_weights('models/' + SELECTED_MODEL + '/parking_agent_target_critic_terminated.h5')

            print(e)