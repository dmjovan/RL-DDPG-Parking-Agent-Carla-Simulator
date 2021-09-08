# ---------------------------------------------------------------------------------------------------
# IMPORTING ALL NECESSARY LIBRARIES
# ---------------------------------------------------------------------------------------------------
import glob
import os
import sys
import time
import numpy as np
import pandas as pd

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
_SLEEP_TIME_ = 1 

FOLDER_PATH = os.getcwd()

# ------------------ DICTIONARY FOR ROW NAMES THAT CORESPONDES TO THE COMMANDS -----------------
df_rows = {

			# corners of goal parking spot - x and y coordinates
			'gdl': 'goal_down_left',
			'gdr': 'goal_down_right',
			'gul': 'goal_upper_left',
			'gur': 'goal_upper_right',

			# orientation of center of goal parking spot - angle in degrees
			'go': 'goal_orientation',

			# spectator row with all fields
			'spec': 'spectator'

		}

# ------------------ CREATING PANDAS DATAFRAME FOR STORING VALUES ----------------------

df_columns = ['x', 'y','z','yaw', 'pitch', 'roll']
df_dtype = 'float32'
df_index = list(df_rows.values())

df = pd.DataFrame(np.nan*np.ones((len(df_index), len(df_columns))))
df.columns = df_columns
df.dtype = df_dtype
df.index = df_index

df.index.name = 'position'

def update_field_in_data_frame(df_row, column = None, value = None):

	"""
    Function for updating field value in Pandas DataFrame, or for storing that
    Pandas DataFrame in .csv file.
        
    :params:
        - df_row: dataFrame row (key in df_rows) for selecting row of field (if 'done', then storing DataFrame in .csv file conducted)
        - column: dataFrame column name for selecting column of field 
        - value: value to be written down in selected field

    :return:
        - done: boolean value, indicating if updating field or storing .csv file is done

    """

	done = False

	if df_row == 'done':
		df.to_csv(FOLDER_PATH + '/parking_map.csv')
		done = True

	elif df_row in df_rows.keys():

		if column in ['yaw', 'pitch', 'roll']:
			value = 360 + value if value < 0 else value

		df.at[df_rows[df_row], column] = value
		done = True

	else:
		print('Wrong input!')
		
	return done

# ---------------------------------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------------------------------

"""
Program for tracking (gathering position info) SPECTATOR in Carla simulator 

Goals of this program - running as independent script:
- getting coordinates of corners and orientation of goal parking spot in world coordinates
- getting coordinates and orientation of spectator camera, to be set in main script for proprior look while training

"""
if __name__ == '__main__':

	client = carla.Client(_HOST_, _PORT_)
	client.set_timeout(10.0)
	world = client.load_world('Town05')

	t = world.get_spectator().get_transform()

	# ------------------ LOOPING WHILE MOVING THROUGH ENVIRONMENT ----------------------
	while(True):


		# ------------------ ROW SELECTION THROUGH SHORT COMMANDS ----------------------

		row = input('Select point: ')

		if row == '':
			continue

		elif row == 'done':
			if update_field_in_data_frame('done'):
				print('Map saved!')
				break
			else:
				print('Error while saving csv file!')
		else:

			# ------------------ GETTING DATA FROM ENVIRONMENT  ----------------------

			t = world.get_spectator().get_transform()

			x = t.location.x
			y = t.location.y
			z = t.location.z
			yaw = t.rotation.yaw
			pitch = t.rotation.pitch
			roll = t.rotation.roll

			# print(x, y, z, yaw, pitch, roll)

			# ---------- UPDATING DATAFRAME'S SELECTED ROW  WITH GATHERED DATA FROM ENVIRONMENT -------------
			
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