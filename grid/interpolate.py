#!/usr/bin/env python3
# Make input data for machine learning of phase separation, by following steps:
# 0. (under bash script and gromacs tool) make 2x2x2 super cell from final coordinate gro file.
#   [ex] gmx genconf -f confout.gro -nbox 2 2 2 -o temp_out.gro
# 1. read 2x2x2 super cell xyz coordinates of A and B particle (for Periodic Boundary condition)
# 2. run (linear) interpolation on grid points instead of off lattice points.
# 3. take only the center of cell
# (option) 4. then copy the cell to make super cell 2x2x2
# 5. translate, rotate, flip up-and-down, and (symmetric) datas to avoid oriental bias for machine learning later.
# 6. save data as being compressed (.npy)
##############
# ver 2.1 - coding python by Hyuntae Jung on 1/31/2019
#           instead of ml_wr.py, we divide several files. 
# ver 2.2 - add #ensembles option on 2/21/2019
# ver 2.3 - add polymer blend model options on 5/30/2019
# ver 3.0 - add three state by adding empty particle on 10/28/2019
import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='input data pre-processing (grid interpolation) of Widom-Rowlinson model')
print(" I suggest to turn off -re_sout and -constr, which may be not necessary for our test.")
## args
parser.add_argument('-i', '--input', default='confout.gro', nargs='?', 
	help='input gro file (will be read only initial coordinate)')
parser.add_argument('-g', '--grid', default=15, nargs='?', type=int, 
	help='number of slices on original cell (not super cell)')
parser.add_argument('-s', '--nsuper', default=2, nargs='?', type=int, 
	help='super cell scaling of input gro file')
parser.add_argument('-super_do', '--super_do', action='store_true', 
	help='make 2x2x2 super cell from input gro file to use grid interpolation (default: OFF)')
parser.add_argument('-itp', '--interp', default='three-states', nargs='?', 
	help='interpolation method (ex. grid/rbf and nearest/linear/cubic, or three-states)')
#  see function selection: 
#   https://stackoverflow.com/questions/32753449/what-to-do-if-i-want-3d-spline-smooth-interpolation-of-random-unstructured-data
#  see method selection:
#   https://stackoverflow.com/questions/51553625/difference-between-scipy-interpolate-griddata-and-scipy-interpolate-rbf
# Thus, even though the same fn gives different result depending interpolation method (grid/rbf)
# I prefer griddata because it clearly gives zeros, not negative values which you often get via rbf method even in all zero area.
# In other words, Rbf shows a large fluctuation beyond a range (like negative values) by neighbors, but better smoothness.
# But, I think either way would be okay.. 
# As Arun requested on 10/22, 
#   interpolation methods such as grid-linear has flaw with empty particles.
#   we are going to put zeros on grid if the zero particles can occupy on grid without no collosion with other A/B particles.
#   We assume that the empty particle size is the grid size.
parser.add_argument('-n_ensembles', '--n_ensembles', default=0, nargs='?', type=int, 
	help='generate # ensembles by trans, flip, and switch axes (randomly #generated up to # you set, but 0 = min.#, negative = max.#)')
parser.add_argument('-debug', '--debug', action='store_true', 
	help='(debug) output gro file for interpolation result')
parser.add_argument('-o', '--output', default='confout.gro', nargs='?', 
	help='output npy file for input file (ex. confout.gro.npy)')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 2.2')
# read args
args = parser.parse_args()
# check args
print(" input arguments: {0}".format(args))

# import modules
import numpy as np
import scipy as sc
import mdtraj as md
import math
import copy 

#np.random.seed(1985)

# set variables
input_gro_file = args.input
n_grid = args.grid
supercell_scale = args.nsuper
re_supercell = args.super_do
output_file = args.output

# read gro file (super cell 2x2x2 
#  to consider periodic boundary condition for interpolation)
print(" ... reading gro file ...")
traj=md.load(input_gro_file)
gro_xyz=traj.xyz[0] # read initial coordinate in time
gro_box=traj.unitcell_lengths[0] # read initial coordinate in time 
if (gro_box[0] != gro_box[1]) or (
	gro_box[0] != gro_box[2]) or (
	gro_box[1] != gro_box[2]):
	raise ValueError("cell dimension of input gro file is not isotropic.")

# make color map
# should be select depending on your model.
print(" ... making color map ...")
## As for WR models,
idx_a = traj.top.select("name A")
idx_b = traj.top.select("name B")
# As for Polymer blends models,
#idx_a = traj.top.select("resname AE or resname AB")
#idx_b = traj.top.select("resname BE or resname BB")

gro_color=np.zeros(traj.top.n_atoms)
#gro_color[idx_b] = 1. # b is white like 1., a is black like 0. for Widom_Rowlinson model
#gro_color[idx_a] = 1. # a is white like 1., b is black like 0. for Polymer_blends
# for new version interpolation, a = white, b=black, empty=gray?
gro_color[idx_a] = 1.  
gro_color[idx_b] = -1. 

print(" ... interpolating ...")
# make mesh grid
x_grid=np.linspace(0,gro_box[0],n_grid*supercell_scale)
#print("n_grid = {}".format(len(x_grid)))
xi, yi, zi = np.meshgrid(x_grid, x_grid, x_grid)
interp_module, interp_fn=(args.interp).split("-")
if 'rbf' in interp_module:
	if not (('cubic' in interp_fn) or ('linear' in interp_fn)):
		raise ValueError("wrong argument args.interp, {}".format(args.interp))
	from scipy.interpolate import Rbf
	# radial basis function interpolator instance

	rbfi = Rbf(gro_xyz[:,0], gro_xyz[:,1], gro_xyz[:,2], gro_color, function=interp_fn, smooth=0.0)
	interp_color = rbfi(xi, yi, zi) # interpolated values, shape=[n_grid*supecell_scale,...,...]
	#print("n_color_interpolated = {}".format(len(interp_color)))
elif 'grid' in interp_module:
	if not (('nearest' in interp_fn) or ('linear' in interp_fn)):
		raise ValueError("wrong argument args.interp, {}".format(args.interp))
	if 'nearest' in interp_fn:
		print("nearest interpolation is not tested yet..")
	from scipy.interpolate import griddata
	interp_color = griddata(gro_xyz, gro_color, (xi, yi, zi), method=interp_fn)
elif 'three' in interp_module:
	grid_size = x_grid[1] - x_grid[0]
	add_xyz = []
	# check if the grid overlaps any A/B particles. 
	#  otherwise, add it to gro_xyz and gro_color to reflect empty space
	for i in np.arange(n_grid*supercell_scale):
		for j in np.arange(n_grid*supercell_scale):
			for k in np.arange(n_grid*supercell_scale):
				trial_xyz=np.array([xi[i,j,k],yi[i,j,k],zi[i,j,k]])
				over_x = np.abs(gro_xyz[:,0] - trial_xyz[0]) < grid_size/2.
				over_y = np.abs(gro_xyz[:,1] - trial_xyz[1]) < grid_size/2.
				over_z = np.abs(gro_xyz[:,2] - trial_xyz[2]) < grid_size/2.
				over_xy = np.logical_and(over_x,over_y)
				over = np.logical_and(over_z,over_xy)
				if int(np.sum(over)) == 0:
					add_xyz.append(trial_xyz)	
				#else:
				#	print("overlapping with {}".format(trial_xyz))

	# adding if there exists
	if len(add_xyz) != 0:
		print("we add {} empty particles".format(len(add_xyz)))
		add_xyz = np.array(add_xyz)
		add_xyz = add_xyz.reshape(-1,3)
		#print(add_xyz)
		add_color = np.repeat(0, len(add_xyz))
		append_xyz = np.append(gro_xyz,add_xyz)
		gro_xyz = append_xyz.reshape(-1,3)
		gro_color = np.append(gro_color,add_color)
	else:
		print("nothing change with grid-linear. Might be not necessary to consider three spin states.")
	from scipy.interpolate import griddata
	interp_color = griddata(gro_xyz, gro_color, (xi, yi, zi), method='linear')
else:
	raise ValueError("the argument keyword does not match, {} -> {} {}".format(args.interp,interp_module,interp_fn))

# remove edges (or nan elements) by slicing
if supercell_scale > 1.:
	print(" ... reducing to original cell ...")
	slice_s = int(n_grid/float(supercell_scale))
	slice_e = slice_s + n_grid
	grid_color=interp_color[slice_s:slice_e,slice_s:slice_e,slice_s:slice_e]
	print(" result (#NaNs): {} -> {}".format(np.sum(np.isnan(interp_color)),np.sum(np.isnan(grid_color))))
	print(" result (#INFs): {} -> {}".format(np.sum(np.isinf(interp_color)),np.sum(np.isinf(grid_color))))
	if (np.sum(np.isnan(grid_color)) > 0.) or (np.sum(np.isinf(grid_color))):
		raise RuntimeError(" Should check why Nan or INF is/are in color map.")

# (optional) make supercell 2x2x2 (to brighthen phase separation)
if re_supercell:
	print(" ... redo make supercell 2x2x2 ...")
	super_grid_color = np.empty([n_grid*supercell_scale,n_grid*supercell_scale,n_grid*supercell_scale])
	# any better way to copy 3d matrix?
	super_grid_color[0:n_grid,0:n_grid,0:n_grid] = copy.copy(grid_color)
	super_grid_color[n_grid:n_grid+n_grid,0:n_grid,0:n_grid] = copy.copy(grid_color)
	super_grid_color[0:n_grid,n_grid:n_grid+n_grid,0:n_grid] = copy.copy(grid_color)
	super_grid_color[0:n_grid,0:n_grid,n_grid:n_grid+n_grid] = copy.copy(grid_color)
	super_grid_color[n_grid:n_grid+n_grid,n_grid:n_grid+n_grid,0:n_grid] = copy.copy(grid_color)
	super_grid_color[n_grid:n_grid+n_grid,0:n_grid,n_grid:n_grid+n_grid] = copy.copy(grid_color)
	super_grid_color[0:n_grid,n_grid:n_grid+n_grid,n_grid:n_grid+n_grid] = copy.copy(grid_color)
	super_grid_color[n_grid:n_grid+n_grid,n_grid:n_grid+n_grid,n_grid:n_grid+n_grid] = copy.copy(grid_color)
	grid_color = copy.copy(super_grid_color)

# make variation coordinates of the cell 
#  by tranlation, flipping, and switching axes.
print(" ... adding variety of copied cells ...")
max_shift=np.shape(grid_color)[0]
if args.n_ensembles == 0:
	# generate smallest # of copied systems
	print(" ... activate least # of copied systems ...")
	n_systems = 1+3+(max_shift-1)*3+3
	data_out = np.empty([n_systems,max_shift,max_shift,max_shift])
	count=0
	data_out[count] = copy.copy(grid_color)
	count+= 1
	# 1. flip
	for i_flip in range(0,3):
		data_out[count] = np.flip(grid_color, i_flip)
		count += 1
	# 2. shift (up to box dimension)
	for j in range(0,3):
		for i in range(1,max_shift):
			data_out[count] = np.roll(grid_color, i, axis=j)
			count += 1
	# 3. switch axes
	data_out[count] = np.moveaxis(grid_color, 0, 1)
	data_out[count+1] = np.moveaxis(grid_color, 0, 2)
	data_out[count+2] = np.moveaxis(grid_color, 1, 2)
	count += 3
	# 4. symmetric
	#data_out[count] = copy.copy(1.0-grid_color)
	#count += 1
elif args.n_ensembles < 0:
	# generate many # copied systems
	print(" ... generate as many as possible ...")
	n_systems = 1+(1+(1+(1+4*max_shift)*max_shift)*max_shift)*3
	#print(n_systems)
	data_out = np.empty([n_systems,max_shift,max_shift,max_shift])
	count=0
	data_out[count] = copy.copy(grid_color)
	count+= 1
	# 1. flip
	for i_flip in range(0,3):
		#print(" ... current count {} ... ".format(count))
		flip_out = np.flip(grid_color, i_flip)
		data_out[count] = copy.copy(flip_out)
		count += 1
		# 2. shift
		for i in range(0,max_shift):
			#print(" ... current count {} ... ".format(count))
			cand1 = np.roll(flip_out, i, axis=0)
			data_out[count] = copy.copy(cand1)
			count += 1
			for j in range(0,max_shift):
				#print(" ... current count {} ... ".format(count))
				cand2 = np.roll(cand1, j, axis=1)
				data_out[count] = copy.copy(cand2)
				count += 1
				for k in range(0,max_shift):
					#print(" ... current count {} ... ".format(count))
					cand3 = np.roll(cand2, k, axis=2)
					data_out[count] = copy.copy(cand3)
					count += 1
					# 3. switch axes
					data_out[count] = np.moveaxis(cand3, 0, 1)
					data_out[count+1] = np.moveaxis(cand3, 0, 2)
					data_out[count+2] = np.moveaxis(cand3, 1, 2)
					count += 3
					#print(i_flip,i,j,k,count)
else:
	# generate many # copied systems
	print(" ... generate fixed #ensembles ...")
	n_systems = args.n_ensembles
	settings=np.array([[0, 0, 0, 0, 0]]) # original
	#make #random setting for flip, shift, switch, symmetric
	# when asking for more than original data
	while len(settings) < n_systems:
		#temp_settings = [rand_order_flip and rand_switch and rand_shift per axes] 
		temp_setting = np.append(np.random.randint(low=0, high=4, size=2),np.random.randint(low=0, high=max_shift+1, size=3))
		try:
			settings = np.vstack((settings, temp_setting))
			#print("else ", settings)
		except ValueError:
			raise RuntimeError("error ", settings, temp_setting)
		settings = np.unique(settings, axis=0)

	if len(settings) != n_systems:
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

	if count != n_systems:
		raise RuntimeError("generating ensembles has problem in size.")

# so many ensembles...? (does it really help?)
if count != n_systems:
	raise ValueError("something wrong loop to get system copies")

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
					#print(i_atom,i_atom,mesh_points[i_atom][0],mesh_points[i_atom][1],mesh_points[i_atom][2],
					#	mesh_colors[i_atom],mesh_colors[i_atom],mesh_colors[i_atom])
					f.write("ATOM  {:>5}  PTS MSH {:>5}     {:>7.3f} {:>7.3f} {:>7.3f}  1.00 {:>5.2f} \n".format(
						i_atom+1,i_atom+1,new_x_grid[ix_pts],new_x_grid[iy_pts],new_x_grid[iz_pts],
						grid_color[ix_pts,iy_pts,iz_pts]*10.))
					i_atom += 1
		f.write("TER \n")
		f.write("ENDMDL \n")
		f.close()


# save file
print("save the file: [{},{},{},{}]".format(n_systems,max_shift,max_shift,max_shift))
np.save(output_file, data_out.flatten())
print("Done: make npy file for interpolation on grid")
