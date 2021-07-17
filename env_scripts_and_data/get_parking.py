# Program for tracking (gathering position info) SPECTATOR in Carla simulator 

# goals of this program - running as independent script>
# - getting coordinates of corners and orientation of goal parking spot in world coordinates
# - getting coordinates of corners and orientation of 2 parking spots on left/right side of goal parking spot (if posible)
# - getting coordinates and orientation of spectator camera, to be set in main script for proprior look while training
# - getting starting coordinates and orientation of vehicle - 2 posible start position 
#   therefore 2 output csv files:
#                                 - parking_map_for_spawn_on_entrance.csv - spawn vehicle on entrance of parking
#                                 - parking_map_for_spawn_on_intersection.csv - spawn vehicle on intersection near parking

import glob
import os
import sys
import time
import numpy as np
import pandas as pd

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla	

_HOST_ = '127.0.0.1' 
_PORT_ = 2000
_SLEEP_TIME_ = 1 # s

# path to the folder
parking_csv_path = os.getcwd()

# dictionary for row names in csv file with keys that corespondes to the commands on the input
# for defining spots while simulating
df_rows = {

            # starting position - x and y coordinates and yaw angle
			's' : 'start',

			# corners of goal parking spot - x and y coordinates
			'dl': 'down_left',
			'dr': 'down_right',
			'ul': 'upper_left',
			'ur': 'upper_right',

			# orientation of center of goal parking spot - angle in degrees
			'co': 'center_orientation',

			# corners of parking spot on the left side of goal parking spot - x and y coordinates
			'ldl': 'left_down_left',
			'ldr': 'left_down_right',
			'lul': 'left_upper_left',
			'lur': 'left_upper_right',

			# orientation of center of parking spot on the left side of goal parking spot - angle in degrees
			'lco': 'left_center_orientation',

			# corners of parking spot on the right side of goal parking spot - x and y coordinates
			'rdl': 'right_down_left',
			'rdr': 'right_down_right',
			'rul': 'right_upper_left',
			'rur': 'right_upper_right',

			# orientation of center of parking spot on the right side of goal parking spot - angle in degrees
			'rco': 'right_center_orientation',

			# spectator row with all fields
			'spec': 'spectator'

		}

# pandas DataFrame to be saved in csv file 
df_columns = ['x', 'y','z','yaw', 'pitch', 'roll']
df_dtype = 'float32'
df_index = list(df_rows.values())

df = pd.DataFrame(np.nan*np.ones((len(df_index), len(df_columns))))
df.columns = df_columns
df.dtype = df_dtype
df.index = df_index

df.index.name = 'position'

# function for storing/updating fields of data frame
def update_field_in_data_frame(df_row, column = None, value = None):

	global starting_waypoint

	done = False

	if df_row == 'done':
		df.to_csv(parking_csv_path + '/parking_map_for_spawn_on_' + starting_waypoint +'.csv')
		done = True

	elif df_row in df_rows.keys():

		if column in ['yaw', 'pitch', 'roll']:

			# return value of angle from 0 to 360 degrees
			value = 360 + value if value < 0 else value

		# updating field
		df.at[df_rows[df_row], column] = value
		done = True

	else:
		print('Wrong input!')
		
	return done

# main program for obataining environment/getting charachteristic spots map
if __name__ == '__main__':

	client = carla.Client(_HOST_, _PORT_)
	client.set_timeout(10.0)
	world = client.load_world('Town05')

	# getting spectators transform
	t = world.get_spectator().get_transform()

	x = t.location.x
	y = t.location.y
	angle = t.rotation.yaw

	starting_waypoint = ''
	waypoint_selected = False

	# loop for choosing the start position for for obtaining
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

	# loop for catching and writing down coordinates
	while(True):

		row = input('Select row: ')

		if row == '':
			continue

		elif row == 'done':
			if update_field_in_data_frame('done'):
				print('Map saved!')
				break
			else:
				print('Error while saving csv file!')
		else:

			# getting spectator transform
			t = world.get_spectator().get_transform()

			x = t.location.x
			y = t.location.y
			z = t.location.z
			yaw = t.rotation.yaw
			pitch = t.rotation.pitch
			roll = t.rotation.roll

			# print(x, y, z, yaw, pitch, roll)

			# updating all fields for one row of data frame
			if not update_field_in_data_frame(row,'x', x):
				print('Error while updating ['+ df_rows[row] +'][x] field in DataFrame!')
			if not update_field_in_data_frame(row,'y', y):
				print('Error while updating ['+ df_rows[row] +'][y] field in DataFrame!')
			if not update_field_in_data_frame(row,'z', z):
				print('Error while updating ['+ df_rows[row] +'][z] field in DataFrame!')
			if not update_field_in_data_frame(row,'yaw', yaw):
				print('Error while updating ['+ df_rows[row] +'][yaw] field in DataFrame!')
			if not update_field_in_data_frame(row,'pitch', pitch):
				print('Error while updating ['+ df_rows[row] +'][pitch] field in DataFrame!')
			if not update_field_in_data_frame(row,'roll', roll):
				print('Error while updating ['+ df_rows[row] +'][roll] field in DataFrame!')

		time.sleep(_SLEEP_TIME_)