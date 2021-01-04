#!/usr/bin/env python3
# ver 0.1 - write code on 04/22/2020

import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='generate augment images for 3rd training set by reading grid.npy file')
## args
parser.add_argument('-i', '--input', default='grid.npy', nargs='?', 
	help='input npy file')
parser.add_argument('-g', '--grid', default=15, nargs='?', type=int, 
	help='number of slices on original cell (not super cell)')
parser.add_argument('-n_files', '--n_files', default=1, nargs='?', type=int, 
	help='# files to generate')
parser.add_argument('-n_ensembles', '--n_ensembles', default=0, nargs='?', type=int, 
	help='# augmented images by trans, flip, and switch axes without original per file')
parser.add_argument('-p', '--prefix', default='grid_3c', nargs='?', 
	help='prefix for output files (grid_3c.$start_index.npy)')
parser.add_argument('-seed', '--seed', default=1985, nargs='?', type=int,
	help='random seed to shuffle for augementation')
parser.add_argument('-debug', '--debug', action='store_true', 
	help='(debug) output gro file for interpolation result')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 3.0')
# read args
args = parser.parse_args()
# check args
print(" input arguments: {0}".format(args))

# import modules
import numpy as np

# set variables
input_npy_file = args.input
n_grid = args.grid
n_files = args.n_files
n_ensembles = args.n_ensembles
prefix=args.prefix
np.random.seed(args.seed)

# load input grid file
in_data = np.load(input_npy_file)
grid_color = in_data.reshape(-1,n_grid,n_grid,n_grid)[0]
print(" ## info of grid.npy ## ")
print(" max and min color  = {}, {}".format(np.max(grid_color),np.min(grid_color)))

# make variation coordinates of the cell 
#  by tranlation, flipping, and switching axes.
print(" ... adding variety of copied cells ...")
max_shift=np.shape(grid_color)[0]
# generate many # copied systems
print(" ... generate fixed #ensembles ...")
n_systems = n_ensembles*(n_files+1)
settings=np.array([[0, 0, 0, 0, 0]]) # original
#make #random setting for flip, shift, switch, symmetric
# when asking for more than original data
while len(settings) < n_systems+1:
	if len(settings)%int(n_systems/10.) == 0:
		print(" ## {} / {} .. {} % ##".format(len(settings),n_systems,int(len(settings)*100/n_systems)))
	temp_setting = np.append(np.random.randint(low=0, high=4, size=2),
		np.random.randint(low=0, high=max_shift+1, size=3))
	try:
		settings = np.vstack((settings, temp_setting))
	except ValueError:
		raise RuntimeError("error ", settings, temp_setting)
	settings = np.unique(settings, axis=0)

# remove original grid color and first n_ensemble 
#  (to avoid overlapping on evalulation set)
settings = settings[1:]
settings = settings[n_ensembles:]
if len(settings) != n_ensembles*n_files:
	raise RuntimeError("generating settings has problem in size.")

# make data_out
data_out = np.empty([n_systems,max_shift,max_shift,max_shift])
count=0
for i_set in settings:
	if i_set[0] != 3:
		flip_out = np.flip(grid_color, i_set[0])
	else:
		flip_out = grid_color
	if i_set[2] != max_shift: 
		cand1 = np.roll(flip_out, i_set[2], axis=0)
	else:
		cand1 = flip_out
	if i_set[3] != max_shift: 
		cand2 = np.roll(cand1, i_set[3], axis=0)
	else:
		cand2 = cand1
	if i_set[4] != max_shift: 
		cand3 = np.roll(cand2, i_set[4], axis=0)
	else:
		cand3 = cand2
	if i_set[1] == 0:
		data_out[count] = np.moveaxis(cand3, 0, 1)
	elif i_set[1] == 1:
		data_out[count] = np.moveaxis(cand3, 0, 2)
	elif i_set[1] == 2:
		data_out[count] = np.moveaxis(cand3, 1, 2)
	else:
		data_out[count] = cand3
	count = count + 1

if count != n_ensembles*n_files:
	raise RuntimeError("generating ensembles has problem in size.")

# (debugging) save pdb file to see color map on grid
if args.debug:
	for i in range(n_systems):
		grid_color = data_out[i]
		print(" ... making {} th debug pdb file ...".format(i))
		f = open(args.input+'.'+str(i)+'.debug.pdb','w')
		f.write("TITLE debug color map on meshgrid \n")
		f.write("CRYST1 {:>8.3f} {:>8.3f} {:>8.3f}  90.00  90.00  90.00 P 1\n".format(
			gro_box[0]/2.,gro_box[0]/2.,gro_box[0]/2.))
		f.write("MODEL        1\n")
		new_x_grid=np.linspace(0,gro_box[0]/2.,(n_grid+1.))
		nx_pts, ny_pts, nz_pts = np.shape(grid_color)
		i_atom = 0
		for ix_pts in range(nx_pts):
			for iy_pts in range(ny_pts):
				for iz_pts in range(nz_pts):
					f.write("ATOM  {:>5}  PTS MSH {:>5}     {:>7.3f} {:>7.3f} {:>7.3f}  1.00 {:>5.2f} \n".format(
						i_atom+1,i_atom+1,new_x_grid[ix_pts],new_x_grid[iy_pts],new_x_grid[iz_pts],
						grid_color[ix_pts,iy_pts,iz_pts]*10.))
					i_atom += 1
		f.write("TER \n")
		f.write("ENDMDL \n")
		f.close()

# save file
print("save each files: [{},{},{},{}]".format(n_ensembles,max_shift,max_shift,max_shift))
for i in range(n_files):
	np.save(prefix+"."+str(i), data_out[i*n_ensembles:(i+1)*n_ensembles].flatten())
print("Done: make npy file for 3rd traning sets, {}".format(prefix))
