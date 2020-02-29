#!/usr/bin/env python3
# to load keras model, run evaluation, and get block averages due to copied ensembles
# ver 3.0 -      
import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='get result of machine learning for evaluation data set')
## args
parser.add_argument('-m', '--model', default='model.h5', nargs='?',  
	help='input ML model file (.h5)')
parser.add_argument('-i', '--input', default='eval', nargs='?',  
	help='prefix of input data file like $input.$nf.(coord/temp).npy')
parser.add_argument('-nf', '--n_files', default=1, nargs='?', type=int,  
	help='# files like $input.$nf.(coord/temp).npy')
parser.add_argument('-ne', '--n_ensembles', default=100, nargs='?', type=int,  
	help='# copied ensembles by trans/flip/reverse')
parser.add_argument('-ng', '--n_grids', default=15, nargs='?', type=int,
	help='# grids in coord.npy data set')
parser.add_argument('-o', '--output', default='result.npy', nargs='?',
	help='output result file (.npy)')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 2.3')
# read args
args = parser.parse_args()
# check args
print(" input arguments: {0}".format(args))

# import module
import copy
import numpy as np
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoid not-risk error messages 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
## start machine learning
import keras

## load model
cnn_model = keras.models.load_model(args.model)

## print out the first layer filter's weights
layer_dict = dict([(layer.name, layer) for layer in cnn_model.layers])
i_layer=0 # first layer
weights=cnn_model.layers[i_layer].get_weights()[0]
nx_pts=weights.shape[0]
ny_pts=nx_pts
nz_pts=nx_pts
bias=cnn_model.layers[i_layer].get_weights()[1]
n_filters=len(bias)
for i in range(n_filters):
	grid_color = weights[:,:,:,:,i].reshape(nx_pts,ny_pts,nz_pts)
	#print(" ... making {} th filter's weights pdb file ...".format(i))
	f = open(args.model+'.filter.'+str(i)+'.pdb','w')
	f.write("TITLE 1st conv layer weights with bias {} \n".format(bias[i]))
	f.write("CRYST1 {:>8.3f} {:>8.3f} {:>8.3f}  90.00  90.00  90.00 P 1\n".format(
		nx_pts*10.,nx_pts*10.,nx_pts*10.))
	f.write("MODEL        1\n")
	new_x_grid=np.linspace(0,nx_pts*10.,nx_pts+1.)
	nx_pts, ny_pts, nz_pts = np.shape(grid_color)
	#print(new_x_grid)
	i_atom = 0
	for ix_pts in range(nx_pts):
		for iy_pts in range(ny_pts):
			for iz_pts in range(nz_pts):
				#print(i_atom,i_atom,mesh_points[i_atom][0],mesh_points[i_atom][1],mesh_points[i_atom][2],
				#	mesh_colors[i_atom],mesh_colors[i_atom],mesh_colors[i_atom])
				f.write("ATOM  {:>5}  PTS MSH {:>5}     {:>7.3f} {:>7.3f} {:>7.3f}  1.00 {:>5.2f} \n".format(
					i_atom+1,i_atom+1,new_x_grid[ix_pts],new_x_grid[iy_pts],new_x_grid[iz_pts],
					grid_color[ix_pts,iy_pts,iz_pts]*50.))
				i_atom += 1
	f.write("TER \n")
	f.write("ENDMDL \n")
	f.close()
print(" done with making filter pdb files")

## load and predict eval. data
n_grids = args.n_grids+2
i_ensemble = 0
for i_file in np.arange(args.n_files):
	# load coord
	print(" load {}-th eval data file to predict Tc".format(i_file))
	coord_iset = np.load(args.input+'.'+str(i_file)+'.coord.npy')
	coord_iset = coord_iset.reshape(-1, n_grids, n_grids, n_grids, 1)
	n_coord_sets = coord_iset.shape[0]
	# determine array size based on file and arguments
	if i_file == 0:
		n_ensembles_per_files = n_coord_sets
		total_n_ensembles = int(n_coord_sets*args.n_files)
		# define arrays
		cat_sets = np.empty(total_n_ensembles)
		temp_sets = np.empty(total_n_ensembles)
	# load temperature
	temp_iset  = np.load(args.input+'.'+str(i_file)+'.temp.npy')
	if temp_iset.shape[0] != n_coord_sets:
		raise ValueError(" inconsistent size for temp_sets with coord_sets, {} != {}".format(
			temp_iset.shape[0], n_coord_sets))
	# prediction
	pred = cnn_model.predict(coord_iset)
	cat_iset = pred[:,1] # predicted category [0:1] are located in 1st column
	#print("size = {} ? {}".format(cat_iset.shape,n_ensembles_per_files))
	#print(temp_iset)
	#print(cat_iset.astype(int))
	# append data
	cat_sets[i_ensemble:i_ensemble+n_ensembles_per_files] = copy.copy(cat_iset)
	temp_sets[i_ensemble:i_ensemble+n_ensembles_per_files] = copy.copy(temp_iset)
	i_ensemble=i_ensemble+n_ensembles_per_files

# swipe memory
del coord_iset
del cat_iset
del temp_iset

# result processing
#print(temp_sets.shape)
#print(temp_sets)
plt_temp = np.unique(temp_sets)
#print(plt_temp)
n_temps  = len(plt_temp)
print(" you have {} temperatures on your data".format(n_temps))
plt_cat_mean = np.empty(n_temps)
plt_cat_std  = np.empty(n_temps)
for i_temp in range(n_temps):
	tmp_array = cat_sets[temp_sets == plt_temp[i_temp]]
	#print("i_temp = {}".format(plt_temp[i_temp]))
	#print("i_temp cat_sets = {}".format(cat_sets))
	tmp_array = tmp_array.reshape(-1,args.n_ensembles)
	avgs = np.average(tmp_array, axis=1) # get average among copied ensembles
	#print("i_temp result = {}".format(avgs))
	plt_cat_mean[i_temp] = np.mean(avgs) 
	plt_cat_std[i_temp] = np.std(avgs)

## save npy file for plotting
out=np.column_stack((plt_temp,plt_cat_mean,plt_cat_std))
np.save(args.output,out)

# done
print("Done: get result of evaluation data")
