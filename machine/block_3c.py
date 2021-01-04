#!/usr/bin/env python3
# clean up data set to simple npy files 
##############
# ver 2.1 - coding python by Hyuntae Jung on 2/3/2019
#           instead of ml_wr.py, we divide several files. 
# ver 2.2 - add n_ensemble option on 2/21/2019
# ver 2.3 - modify for 3 classes on 8/13/2019 
import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='generate data block for machine learning input')
## args
parser.add_argument('-i', '--input', default='target.list', nargs='?', 
	help='input list file (format $file_index $temperature/density)')
parser.add_argument('-ipf', '--input_prefix', default='grid', nargs='?',  
	help='prefix of input grid .npy file')
parser.add_argument('-s1', '--select1', default=0.5, nargs='?', type=float, 
	help='select temperature/density1 (< args.select2) for training set')
parser.add_argument('-s2', '--select2', default=1.0, nargs='?', type=float, 
	help='select temperature/density2 (> args.select1) for training set')
parser.add_argument('-s3', '--select3', default=-1, nargs='?', type=float, 
	help='select temperature/density3 (select1 < T < select2) for training set')
parser.add_argument('-prop', '--prop', default=-1.0, nargs='?', type=float, 
	help='the proportion [0:1] of training set for getting accuracy of modeling (< 0. means nothing test set)')
parser.add_argument('-nb', '--n_blocks', default=0, nargs='?', type=int,
	help='# of blocks for training set (zero means no block average sets)')
parser.add_argument('-nbe', '--n_blocks_eval', default=0, nargs='?', type=int,
	help='# of blocks for eval set (due to reduced file size) (zero means no block average sets)')
parser.add_argument('-net', '--ne_train', default=-1, nargs='?', type=int,
	help='# of ensembles for train set per grid.npy (-1 to use all)')
parser.add_argument('-nee', '--ne_eval', default=-1, nargs='?', type=int,
	help='# of ensembles for eval set per grid.npy (-1 to use all)')
parser.add_argument('-ng', '--n_grids', default=15, nargs='?', type=int,
	help='# of grids for data sets')
parser.add_argument('-seed', '--seed', default=1985, nargs='?', type=int,
	help='random seed to shuffle for test sets and block sets')
parser.add_argument('-o1', '--out_train', default='train', nargs='?',
	help='prefix of output training set .npy file like train.(coord/temp/cat).$i.npy')
parser.add_argument('-o2', '--out_test', default='test', nargs='?',
	help='prefix of output test set .npy file for accuracy like test.(coord/temp/cat).npy')
parser.add_argument('-o3', '--out_eval', default='eval', nargs='?',
	help='prefix of output Tc evaluation set .npy file like eval.(coord/temp).npy')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 2.2')
# read args
args = parser.parse_args()
# check args
print(" input arguments: {0}".format(args))

# import modules
import numpy as np
import scipy as sc
import math
import copy 
np.random.seed(args.seed)

# step1: read list file and split to train, test, and eval sets.
list_file = np.loadtxt(args.input)
list_temp = np.array(list_file[:,0],dtype=float)
list_file_idx = np.array(list_file[:,1],dtype=int)
train_set1 = np.where(list_temp == args.select1)[0] # indices for temp1 of training
train_set2 = np.where(list_temp == args.select2)[0] # indices for temp2 of training
train_set3 = np.where(list_temp == args.select3)[0] # indices for temp3 of training
eval_set = np.delete(np.arange(len(list_file_idx)), np.append(train_set1,np.append(train_set2,train_set3))) # indices for eval

# make train_set and test_set with proportion and shuffle
if args.prop > 0.0:
	if args.prop >= 0.5:
		raise ValueError("args.prop {} is too high unlike purpose".format(args.prop))
	n_test1 = int(len(train_set1)*args.prop)
	n_test2 = int(len(train_set2)*args.prop)
	n_test3 = int(len(train_set3)*args.prop)
	np.random.shuffle(train_set1)
	np.random.shuffle(train_set2)
	np.random.shuffle(train_set3)
	test_set = np.append(train_set1[0:n_test1],np.append(train_set2[0:n_test2],train_set3[0:n_test3]))
	train_set1 = train_set1[n_test1:]
	train_set2 = train_set2[n_test2:]
	train_set3 = train_set3[n_test3:]
else:
	print(" Not make test set")
	np.random.shuffle(train_set1)
	np.random.shuffle(train_set2)
	np.random.shuffle(train_set3)
	test_set = np.array([],dtype=int)

print("Based on {} list file: ".format(args.input))
print(" total #train data: {} for temp == {}, {} for temp == {}, {} for temp == {}".format(
	len(train_set1),args.select1,len(train_set2),args.select2,len(train_set3),args.select3))
print(" #test data: {}".format(len(test_set)))
print(" #eval data: {}".format(len(eval_set)))

# step2: make blocks for training sets.
if args.n_blocks > 0:
	remain_1 = len(train_set1)%args.n_blocks
	remain_2 = len(train_set2)%args.n_blocks
	remain_3 = len(train_set3)%args.n_blocks
	print(" trim ({},{}) elements from two training sets for equal size of block sets".format(remain_1,remain_2))
	if remain_1 > 0:
		train_set1 = train_set1[remain_1:]
	if remain_2 > 0:
		train_set2 = train_set2[remain_2:]
	if remain_3 > 0:
		train_set3 = train_set3[remain_3:]
	block_sets1 = np.split(train_set1,args.n_blocks)
	block_sets2 = np.split(train_set2,args.n_blocks)
	block_sets3 = np.split(train_set3,args.n_blocks)
	print(" #blocks for training set = {}".format(args.n_blocks))
else:
	print(" no blocks for training sets")
	block_sets1 = train_set1
	block_sets2 = train_set2
	block_sets3 = train_set3

# step3: make blocks for evaluation sets:
if args.n_blocks_eval > 0:
	if len(eval_set)%args.n_blocks_eval != 0 :
		raise ValueError("n_blocks_eval value is not good to splite eval_set ({} % {} != 0)".format(len(eval_set),args.n_blocks_eval))
	block_sets_eval = np.split(eval_set,args.n_blocks_eval)
	print(" #blocks for eval set = {}".format(args.n_blocks_eval))
else:
	print(" no blocks for eval sets")
	block_sets_eval = eval_set

# without padding
def make_npy_files_mode_ver0(mode, i_block, idx_array, input_prefix, output_prefix):
	# mode = test/eval/train
	if ("test" in mode) or ("train" in mode):
		gen_cat = True 
	else:
		gen_cat = False # eval case
	# initialzie arrays
	# As for eval set, we only use original grid info excluding ensembles or copies by trans, rot, and flip.
	n_data = len(idx_array)
	if gen_cat:
		esti_n_sets = args.ne_train
		set_coord=np.empty((n_data,esti_n_sets*pow(args.n_grids,3)))
		set_temp=np.empty((n_data,esti_n_sets))
		set_cat=np.empty((n_data,esti_n_sets))
	else: # eval case
		esti_n_sets = args.ne_eval
		set_coord=np.empty((n_data,esti_n_sets*pow(args.n_grids,3)))
		set_temp=np.empty((n_data,esti_n_sets))
	print(" collecting sets for {} mode".format(mode))
	# run each sample
	for i_data in np.arange(n_data):
		# load data
		i_set = list_file_idx[idx_array[i_data]]
		filename = input_prefix+"."+str(i_set)+".npy"
		try:
			tmp_data = np.load(filename)
		except FileNotFoundError:
			raise ValueError("{} file does not found. Please remove the filename in list file".format(filename))
		# check #ensembles
		n_sets=int(len(tmp_data)/args.n_grids/args.n_grids/args.n_grids)
		if (esti_n_sets != n_sets) and gen_cat:
			raise RuntimeError("#ensembles sizes are different in {} file like {} != {}".format(filename, n_ensembles, n_sets))
		# assign coord data
		if gen_cat:
			set_coord[i_data]=copy.copy(tmp_data)
		else:
			#set_coord[i_data]=copy.copy(tmp_data[0:pow(args.n_grids,3)]) # for single ensemble
			set_coord[i_data]=copy.copy(tmp_data[0:pow(args.n_grids,3)*n_eval_ensembles])
		# assign cat and temp data
		tmp_temp = list_temp[idx_array[i_data]]
		if gen_cat:
			if tmp_temp <= args.select1:
				set_cat[i_data]=np.repeat(0.,esti_n_sets) # mixed
			elif tmp_temp >= args.select2: 
				set_cat[i_data]=np.repeat(1.,esti_n_sets) # separation
			else:
				raise ValueError("mixed or seperated? see temperature {} != ({} or {})".format(
					tmp_temp, args.select1, args.select2))
		set_temp[i_data]=np.repeat(tmp_temp,esti_n_sets)
	# save compressed npy files
	if i_block is None:
		np.save(output_prefix+'.coord', set_coord.flatten())
		np.save(output_prefix+'.temp', set_temp.flatten())
		if gen_cat:
			np.save(output_prefix+'.cat', set_cat.flatten())
		print("#{} samples = {}".format(mode, n_data))
	else:
		np.save(output_prefix+'.'+str(i_block)+'.coord', set_coord.flatten())
		np.save(output_prefix+'.'+str(i_block)+'.temp', set_temp.flatten())
		if gen_cat:
			np.save(output_prefix+'.'+str(i_block)+'.cat', set_cat.flatten())
		print("#{} {} samples = {}".format(mode, i_block, n_data))

# with PBC padding	
def make_npy_files_mode(mode, i_block, idx_array, input_prefix, output_prefix):
	# mode = test/eval/train
	if ("test" in mode) or ("train" in mode):
		gen_cat = True 
	else:
		gen_cat = False # eval case
	# initialzie arrays
	# As for eval set, we only use original grid info excluding ensembles or copies by trans, rot, and flip.
	n_data = len(idx_array)
	if gen_cat:
		esti_n_sets = args.ne_train
		set_coord=np.empty((n_data,esti_n_sets*pow(args.n_grids+2,3)))
		set_temp=np.empty((n_data,esti_n_sets))
		set_cat=np.empty((n_data,esti_n_sets))
	else: # eval case
		esti_n_sets = args.ne_eval
		set_coord=np.empty((n_data,esti_n_sets*pow(args.n_grids+2,3)))
		set_temp=np.empty((n_data,esti_n_sets))
	print(" collecting sets for {} mode".format(mode))
	# run each sample
	for i_data in np.arange(n_data):
		# load data
		i_set = list_file_idx[idx_array[i_data]]
		filename = input_prefix+"."+str(i_set)+".npy"
		try:
			tmp_data = np.load(filename)
		except FileNotFoundError:
			raise ValueError("{} file does not found. Please remove the filename in list file".format(filename))
		# check #ensembles
		n_sets=int(len(tmp_data)/args.n_grids/args.n_grids/args.n_grids)
		if esti_n_sets > n_sets:
			raise RuntimeError("#ensembles sizes you asked are less than #sets in {} file like {} > {}".format(filename, esti_n_sets, n_sets))
		tmp_data = tmp_data.reshape(n_sets,args.n_grids,args.n_grids,args.n_grids)
		# add padding on tmp_data for each ensemble
		# if load_input_eval_file has more ensembles than esti_n_sets, only save first esti_n_sets data in block files
		for esti_i_sets in range(esti_n_sets):
			tmp_org = tmp_data[esti_i_sets]
			tmp_pad1 = np.empty((args.n_grids+2,args.n_grids,args.n_grids)) # add yz layer
			for ix in range(args.n_grids+2):
				tmp_pad1[ix] = tmp_org[(ix-1)%args.n_grids]
			tmp_pad2 = np.empty((args.n_grids+2,args.n_grids+2,args.n_grids)) # add xz layer
			for iy in range(args.n_grids+2):
				tmp_pad2[:,iy] = tmp_pad1[:,(iy-1)%args.n_grids]
			tmp_pad3 = np.empty((args.n_grids+2,args.n_grids+2,args.n_grids+2)) # add xz layer
			for iz in range(args.n_grids+2):
				tmp_pad3[:,:,iz] = tmp_pad2[:,:,(iz-1)%args.n_grids]
			# assign coord data
			start_idx=esti_i_sets*pow(args.n_grids+2,3)
			end_idx=(esti_i_sets+1)*pow(args.n_grids+2,3)
			set_coord[i_data,start_idx:end_idx]=copy.copy(tmp_pad3.flatten())
		# assign cat and temp data
		tmp_temp = list_temp[idx_array[i_data]]
		if gen_cat:
			if tmp_temp == args.select1:
				set_cat[i_data]=np.repeat(0.,esti_n_sets) # select1
			elif tmp_temp == args.select2: 
				set_cat[i_data]=np.repeat(1.,esti_n_sets) # select2
			elif tmp_temp == args.select3: 
				set_cat[i_data]=np.repeat(2.,esti_n_sets) # select3	(p surface)
			else:
				raise ValueError("mixed or seperated? see temperature {} != ({}, {}, or {})".format(
					tmp_temp, args.select1, args.select2, args.select3))
		set_temp[i_data]=np.repeat(tmp_temp,esti_n_sets)
	# save compressed npy files
	if i_block is None:
		np.save(output_prefix+'.coord', set_coord.flatten())
		np.save(output_prefix+'.temp', set_temp.flatten())
		if gen_cat:
			np.save(output_prefix+'.cat', set_cat.flatten())
		print("#{} samples = {}".format(mode, n_data))
	else:
		np.save(output_prefix+'.'+str(i_block)+'.coord', set_coord.flatten())
		np.save(output_prefix+'.'+str(i_block)+'.temp', set_temp.flatten())
		if gen_cat:
			np.save(output_prefix+'.'+str(i_block)+'.cat', set_cat.flatten())
		print("#{} {} samples = {}".format(mode, i_block, n_data))


# step3: make .npy files for train, test, and eval with blocks.
# test_set 
if len(test_set) > 0:
	make_npy_files_mode("test", None, test_set, args.input_prefix, args.out_test)
else:
	print(" not generated test set output")
# eval set
if args.n_blocks_eval > 0:
	print(" collecting block sets for eval")
	for i_block in range(args.n_blocks_eval):
		print(" ... {}th block ... ".format(i_block))
		make_npy_files_mode("eval", i_block, block_sets_eval[i_block], args.input_prefix, args.out_eval)
else:
	print(" collecting total (no block) sets for eval")
	make_npy_files_mode("train", None, block_sets_eval, args.input_prefix, args.out_train)

# training set
if args.n_blocks > 0:
	print(" collecting block sets for training")
	for i_block in range(args.n_blocks):
		print(" ... {}th block ... ".format(i_block))
		tmp_set = np.append(block_sets1[i_block],np.append(block_sets2[i_block],block_sets3[i_block]))
		make_npy_files_mode("train", i_block, tmp_set, args.input_prefix, args.out_train)
else:
	print(" collecting total (not block) sets for training")
	tmp_set = np.append(block_sets1,block_sets2,block_sets3)
	make_npy_files_mode("train", None, tmp_set, args.input_prefix, args.out_train)

print("Done: make data sets for machine learning")
