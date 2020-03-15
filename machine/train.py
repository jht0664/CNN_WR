#!/usr/bin/env python3
# Do machine learning depending mode train/test/eval 
#  Anyway, this work was inspired by a paper published in Nature Physics 2017; 
#  "Machine learning phases of matter" by Juan Carrasquilla & Roger G. Melko
#  Nature Physics volume 13, pages 431–434 (2017)
#  https://www.nature.com/articles/nphys4035
##############
# ver 2.1 - coding python by Hyuntae Jung on 2/3/2019
#           instead of ml_wr.py, we divide several files. 
# ver 2.2 - add test mode for neural network layers on 2/4/2019
# ver 2.3 - move the making plots into another python script (plot.py) on 2/16/2019          
# ver 3.0 - divide into two jobs; making models (machine3.py) and make result of evaluation data(eval_result.py)
# ver 3.1 - add pbc boundary for "valid" padding 
#			and save opt. CNN filters for reporting characteristics of phase separation on 3/27/2019
# ver 3.3 - add random_seed
# ver 3.3 - limit the usage for Widom-Rowlinson model
import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='supervised machine learning for phase separation of Widom-Rowlinson model')
## args
parser.add_argument('-i', '--input', default='train.0', nargs='?',  
	help='prefix of input .npy train file like $input.$i.(coord/temp/cat).npy')
parser.add_argument('-it', '--input_test', default='NONE', nargs='?',  
	help='prefix of input .npy test file like $input_test.(coord/temp/cat).npy, otherwise put NONE.')
parser.add_argument('-ng', '--n_grids', default=15, nargs='?', type=int,
	help='# grids in input_prefix.coord.npy ')
parser.add_argument('-config', '--config_model', default='model.config', nargs='?',
	help='test mode for the structure of network layers (format: dropout \n conv1 \n conv2 \n pool \n dense*)')
parser.add_argument('-seed', '--seed', default=-1, nargs='?', type=int,
	help='set random seed (negative or zero value means random without seed)')
parser.add_argument('-o', '--output', default='model.h5', nargs='?',
	help='output network model file (.h5)')

parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 3.1')
# read args
args = parser.parse_args()
# check args
print(" input arguments: {0}".format(args))

# import module
import numpy as np
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoid not-risk error messages 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 
## start machine learning
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D

if args.seed > 0:
	from numpy.random import seed
	seed(args.seed)
	from tensorflow import set_random_seed
	set_random_seed(args.seed+1)

n_grids = args.n_grids + 2 # due to PBC padding 

## load train data
print(" load train data with prefix {}".format(args.input))
train_coord_sets = np.load(args.input+'.coord.npy')
train_coord_sets = train_coord_sets.reshape(-1, n_grids, n_grids, n_grids, 1)
n_sets = train_coord_sets.shape[0]

train_cat_sets = np.load(args.input+'.cat.npy')
if train_cat_sets.shape[0] != n_sets:
	raise ValueError(" inconsistent size for cat_sets with coord_sets, {} != {}".format(
		train_cat_sets.shape[0],n_sets))
train_cat_sets = keras.utils.to_categorical(train_cat_sets, 2)

train_temp_sets  = np.load(args.input+'.temp.npy')
if train_temp_sets.shape[0] != n_sets:
	raise ValueError(" inconsistent size for temp_sets with coord_sets, {} != {}".format(
		train_temp_sets.shape[0],n_sets))

## prepare for CNN input layer and out layer
## modeling (construct CNN layers)
#  see details: 
#   https://liufuyang.github.io/2017/04/01/just-another-tensorflow-beginner-guide-3.html
#   http://cs231n.github.io/convolutional-networks/
input_shape = (n_grids, n_grids, n_grids, 1)

# original model for preliminary result of WR model
def modeling_ver0():
	model = Sequential()
	# first hidden layer;
	#  32 feature maps (or filter), which with the filter size of 3x3x3 
	#  and a rectifier activation function 
	# Note that the CONV layer’s parameters consist of a set of learnable filters
	model.add(Conv3D(32, kernel_size=(3, 3, 3),
					strides=(1, 1, 1), padding='same',
					activation='relu', input_shape=input_shape)) # activate by higher size, 32 -> 64
	# second hidden layer;
	#  64 feature maps, which with the size of 3x3x3 
	#  and a rectifier activation function 
	model.add(Conv3D(64, (3, 3, 3), 
					strides=(1, 1, 1), padding='same', 
					activation='relu'))
	# pooling layer
	#  with pool size of 2x2x2
	#   which means with a stride of 2 downsamples every depth slice in the input by 2 
	#   along both width and height, discarding 75% of the activations
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))
	# a regularization layer using dropout 
	#  andomly exclude 20% of neurons in the layer 
	#  in order to reduce overfitting.
	model.add(Dropout(0.2))
	# converts the 3D matrix data to a vector
	#  It allows the output to be processed 
	#   by standard fully connected layers
	model.add(Flatten())
	# a fully connected layer with 128 neurons
	#  and rectifier activation function.3
	#  How to decide #nuerons? see https://www.heatonresearch.com/2017/06/01/hidden-layers.html
	#   1. The number of hidden neurons should be between the size of the input layer and the size of the output layer.
	#   2. The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
	#   3. The number of hidden neurons should be less than twice the size of the input layer.
	#  from (1), should be [2:15*15*15] = [2:3375]
	#  from (2), should be 15*15*15*2/3+2 ~ 2048
	#  from (3), should be < 15*15*15*2  
	model.add(Dense(128, activation='relu'))
	# As there will be many weights generated on the previous layer, 
	#  it is configured to randomly exclude 30% of neurons in the layer 
	#  in order to reduce overfitting.
	model.add(Dropout(0.3))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.1))
	# the output layer has 2 neurons for the 2 classes 
	#  and a softmax activation function 
	#   to output probability-like predictions for each class
	model.add(Dense(2, activation='softmax'))
	model.summary()
	model.compile(loss='binary_crossentropy',
	              optimizer="adam",
	              metrics=['accuracy'])
	return model

def user_model(config_file):
	print("construct machine learning model by {} file".format(config_file))
	model = Sequential()
	config_array = []
	config=open(config_file, 'r')
	# read dropout probablity
	prob_drop = float(config.readline().split()[0])
	if (prob_drop > 1.0) or (prob_drop < 0.0):
		raise ValueError(" prob_drop is too big or small, {}".format(prob_drop))
	elif prob_drop == 0.:
		bool_drop = False
		print(" deactivate dropout fn")
	else:
		bool_drop = True
		print(" activate dropout fn")
	# read 1st conv. layer
	filter_size, n_conv = np.int_(config.readline().split()[0:2])
	if n_conv > 0:
		model.add(Conv3D(n_conv, kernel_size=(filter_size, filter_size, filter_size),
					strides=(1, 1, 1), padding='valid',
					activation='relu', input_shape=input_shape))
		print(" add 1st conv3D layer {}".format(n_conv))
		config_array.append("conv "+str(n_conv))
	else:
		raise ValueError(" wrong value for 1st n_conv, {}".format(n_conv))
	# read 2nd conv. layer
	filter_size, n_conv = np.int_(config.readline().split()[0:2])
	if n_conv > 0:
		model.add(Conv3D(n_conv, kernel_size=(filter_size, filter_size, filter_size),
					strides=(1, 1, 1), padding='valid',
					activation='relu'))
		print(" add 2nd conv3D layer {}".format(n_conv))
		config_array.append("conv "+str(n_conv))
	elif n_conv == 0:
		print(" pass 2nd conv3D layer")
	else:
		raise ValueError(" wrong value for 2nd n_conv, {}".format(n_conv))
	# read avg. pooling layer
	n_stride = int(config.readline().split()[0])
	if n_stride > 0:
		model.add(AveragePooling3D(pool_size=(n_stride, n_stride, n_stride)))
		print(" add ave. pooling layer")
		config_array.append("pool "+str(n_stride))
		if bool_drop:
			model.add(Dropout(prob_drop))
			config_array.append("dropout "+str(prob_drop))
	elif n_stride == 0:
		print(" pass avg. pooling layer")
	else:
		raise ValueError(" wrong value for max. pooling layer, {}".format(n_conv))
	# fully connected arrays
	model.add(Flatten())
	# read dense layers (exclude output layer)
	tmp = config.readlines()
	n_dense = len(tmp)
	for i in range(n_dense):
		try:
			n_neurons=int(tmp[i].split()[0])
		except IndexError:
			raise IndexError("Probably you put whitespace somewhere in the file, {}".format(tmp))
		if n_neurons != 0:
			model.add(Dense(n_neurons, activation='relu'))
			print(" add Dense layer {}".format(n_neurons))
			config_array.append("Dense "+str(n_neurons))
			if bool_drop:
				model.add(Dropout(prob_drop))
				config_array.append("dropout "+str(prob_drop))
		else:
			continue
	if n_dense == 0:
		print(" pass any Dense layer")
	# add output layer
	model.add(Dense(2, activation='softmax'))
	model.summary()
	print("config model = {}".format(config_array))
	model.compile(loss='binary_crossentropy',
	              optimizer="adam",
	              metrics=['accuracy'])
	return model

# fiting with training sets
#cnn_model = modeling_ver0() # 1,082 k parameters: 74s/epoch, 3 epoch -> 97.4% accuracy, 4 epoch -> 99.7%
cnn_model = user_model(args.config_model)
history = cnn_model.fit(train_coord_sets, train_cat_sets,
                    batch_size=50,
                    epochs=30,
                    verbose=1,
                    shuffle=True)
print("Done by fitting to training set")
# release memory 
del train_coord_sets
del train_cat_sets
del train_temp_sets

## load test data if available
if 'NONE' not in args.input_test:
	print(" load test data with prefix {} to calculate accuracy".format(args.input_test))
	try:
		test_coord_sets = np.load(args.input_test+'.coord.npy')
		test_coord_sets = test_coord_sets.reshape(-1, n_grids, n_grids, n_grids, 1)
		n_sets = test_coord_sets.shape[0]
		test_cat_sets = np.load(args.input_test+'.cat.npy')
		if test_cat_sets.shape[0] != n_sets:
			raise ValueError(" inconsistent size for cat_sets with coord_sets, {} != {}".format(
				test_cat_sets.shape[0],n_sets))
		test_cat_sets = keras.utils.to_categorical(test_cat_sets, 2)

		test_temp_sets  = np.load(args.input_test+'.temp.npy')
		if test_temp_sets.shape[0] != n_sets:
			raise ValueError(" inconsistent size for temp_sets with coord_sets, {} != {}".format(
				test_temp_sets.shape[0],n_sets))
		# check results with test sets
		score = cnn_model.evaluate(test_coord_sets, test_cat_sets, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		del test_coord_sets
		del test_temp_sets
		del test_cat_sets
	except IOError:
		print("not found the file. Skip test data")
		args.input_test="NONE"
		pass
else:
	print(" No test for accuracy")

# save network model
cnn_model.save(args.output)

print("Done: construct machine learning model")
