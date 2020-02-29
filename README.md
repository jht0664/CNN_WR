# CNN_WR
Convolutional Neural Network prediction for the phase behavior of three-dimensional Widom-Rowlinson (WR) Model

# Tutorial from A to Z
## A. generate an initial coordinate
### move to the tutorial folder 
```
> cd tutorial
```
### run generator for initial coordinate of random configuration of WR particles;
 in this example, the program tries to insert 512 *A* particles and 512 *B* particles (radius = 1.0 *sigma*) 
 to satisfy number density 0.5 up to 1,000,000 insertion attempts

output text file, `init.ic`, will be used for Monte Carlo simulation inpute file
```
> python ../init_mc/gen_init.py -na 512 -nb 512 -d 0.55 -r 1 -mt 100000
> mv init.ic composite.ic
```
You can see the text file, `composite.ic`, for initial coordinate.

Note that positions should be saved in double-precision for accuracy. 

### Run Monte Carlo (MC) Simulation to equilibrate system

For the Fortran 95 version of the program, take a look at `README` file in `../init_mc/mcrun_v2.tar`
```
> ../init_mc/mcrun.x
```
intput files: `composite.ic, mcrun.inp` | output files: `composite.tmp, composite.fc, conf.gro, confout.gro, mcrun.out` 

While we run MC with double precision coordinates, 
 we take equilibrium system with single-precision file, confout.gro, for next step.

## B. Grid interpolation
### Generate SuperCell
We are going to interpolate 3D coordinates of WR into 3D grid matrix (or lattice, 3D image).
Currently, the built-in grid interpolation function in Python Scipy library has **NaN** issue on edge interpolation.
To aovid this, we need a **super cell** with a factor of 2 on axis or (2 x 2 x 2) with copied coordinates.

When applying grid-interpolation on super cell, we remove edges off, then only 1 x 1 x 1 cell 3d image will be left.
In details of method, please see my article. 
For convenience, I upload `super.gro` file for the supercell coordinate.

### Run grid-interpolation program;

For this case, we will assign 3d coordinates on 10 x 10 x 10 grids (to be close to total number of particles, N=1024),
 and augment the 3d image by a factor of 100 using transformations (translation, replacement of axis, and flipping, etc).
```
> python ../grid/interpolate.py -i super.gro -g 10 -s 2 -itp three-states -n_ensembles 100 -o grid
```
input file: `super.gro` | output file: `grid.npy` -> size (100 x 10 x 10 x 10) ensemble x grid x grid x grid

## C. Convolutional Neural Network (CNN)

### Make dataset as ML inputs

To do find phase boundary at a fixed concentration, you need to manipulate many grid.npy files at different densities. 
When you prepare those `grid.${idx}.npy` files with a variety of density, 
 make a list of files in target.list which has two columns; density {tab} grid file index `grid.$idx.npy`.
Then, make dataset files for training, test, and evaluation data, shuffling with random seed *1985*.

The training dataset should contain 3d images at two densities; 0.5 and 1.0. Rest of 3d images at other densities will be in evaluation file. Also, I employed periodic boundary condition padding (1x1x1) to avoid edge effect when training ML models. Thus, the final 3d images is 12 x 12 x 12 by adding paddings on 6 faces; left, right, top, bottom, front, and back. 
```
> python ~/new_WR/script/machine/block.py -i target.list -ipf grid -s1 0.5 -s2 1.0 -prop 0.0 -nb 1 -seed 1985 -ng 10 -nbe 1 -ne 1
```
input file: `target.list, grid.0.npy, grid.1.npy, ..., grid.5100.npy` | output file: `train.0.cat.npy, train.0.coord.npy, train.0.temp.npy, eval.0.coord.npy, eval.0.temp.npy`


### CNN network model training

For the sake of time, I upload my dataset on Google Drive for while.
In this tutorial, we are going to use datasets at 0.125 concentration with total 2048 particles.  
Please download `conc0.125_n2048` folder in the following link of Google Drive.
Then, move the 7 files to your `tutorial` folder.
Note that any folder size would be at least from 300 MB to 2.0 GB. 

https://drive.google.com/drive/folders/12hfoGuFf3DwGLALJ9O-m7221JZTT7ozE?usp=sharing

Again, from this part, we are going to start N=2048 system (#grid = 13) unlike previous example N=1024 system.
The input file, `model.config' consists of lines to construct our neural network model;
```
(in tutorial folder)
> cat model.config
0.2
2 16
2 16
0
256
```
Adding comments on formatting of model.config file: 1st line is for probability of dropout layer, 20 % data will be dropped after every dense layer (or linear classifier layer).
2nd line means for feature size (2 x 2 x 2) with 16 features for 1st CNN layer. The feature size is changable to find the optimal size. 
3rd line indicates the feature size (2 x 2 x 2) with 16 features for 2nd CNN layer. This line should be kept fixed to check our developed CNN scheme. (see my article).  
4th line is stride size for average pooling layer after 2nd CNN layer. Zero means no pooling layer after 2nd CNN layer.
Starting with 5th lines, the scores after 4th line will be a fully connected (FC) layer. Then, we can set several dense layer(s) except binary classification (output) layer.
5th line shows the number of neurons in 1st dense layer for FC layer. We used 256 neurons for 1st dense layer
Next i-th lines will give additional dense layers.
No more line will stop constructing neural network by adding output layer with SoftMax activation function.
I refer my article about details of CNN model.

It is the time to train our ML model. Note that the seed in argument is used to randomly initialize weights and biases. Although 3d images of input .npy files has the size of 15 x 15 x 15 including padding, I already consider adding sizes with 13 x 13 x 13 grids in argument.  

```
> python ../machine/train.py -i train.0 -ng 13 -config model.config -seed 1985 -o model.h5
construct machine learning model by model.config.0 file
 activate dropout fn
 add 1st conv3D layer 16
 add 2nd conv3D layer 16
 pass avg. pooling layer
 add Dense layer 256
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv3d_1 (Conv3D)            (None, 14, 14, 14, 16)    144
_________________________________________________________________
conv3d_2 (Conv3D)            (None, 13, 13, 13, 16)    2064
_________________________________________________________________
flatten_1 (Flatten)          (None, 35152)             0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               8999168
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 514
=================================================================
Total params: 9,001,890
Trainable params: 9,001,890
Non-trainable params: 0
_________________________________________________________________
config model = ['conv 16', 'conv 16', 'Dense 256', 'dropout 0.2']
Epoch 1/30

  200/20000 [..............................] - ETA: 4:13 - loss: 0.7034 - acc: 0.4700
  800/20000 [>.............................] - ETA: 1:02 - loss: 1.6783 - acc: 0.5075
(...)
Using TensorFlow backend.
Done by fitting to training set
 load test data with prefix test to calculate accuracy
not found the test file. Skip test data
Done: construct machine learning model
```

input file: `train.0.coord.npy, train.0.temp.npy, train.0.cat.npy` for a set of 3d images, density of each image, and category class of each image | output file: `model.h5` for weights and bias for our ML model in the format of .h5.

### Phase prediction on evaluation dataset

This program support phase prediction with some dataset in serial to avoid loading large file size. 
Because I split all evaluation data to two files, we have two differen files, `eval.0.${coord/temp}.npy` and `eval.1.${coord/temp}.npy`.
Using `-nf 2` for two files and `-ne` and `-ng` to determine the input data size, 
 we can predict the phase transition at a given concentration.
```
> python ../machine/predict.py -m model.h5 -i eval -nf 2 -ne 1 -ng 13 -o model.result
 input arguments: Namespace(args=[], input='eval', model='model.h5', n_ensembles=1, n_files=2, n_grids=13, output='model.result')
Using TensorFlow backend.
 done with making filter pdb files
 load 0-th eval data file to predict Tc
 load 1-th eval data file to predict Tc
 you have 49 temperatures on your data
Done: get result of evaluation data
```
input files: `eval.0.{coord/temp}.npy and eval.1.{coord/temp}.npy` | output files: `model.result.npy, model.h5.filter.{0-15}.pdb` 
`model.h5.filter.{0-15}.pdb` are the .gro file for feature 3d map with optimized weight and bias. Use VMD software to visualize.

## D. Estimation of phase transition point

### Fit sigmoid function 
We use the probability of phase w.r.t density in `model.result.npy`, then fit with sigmoid function.
The density that has 50% probability of phase will be phase transition point; as you can see below, 
 transition point is 0.82357 for N=2048 system with 0.125 concentration. 
 To check fitting graph and data points, open `model.result.png` in image viewer.

```
> python ../machine/plot.py -i model.result.npy -o model.result.png
 input arguments: Namespace(args=[], criteria=0.5, input='model.result.npy', output='model.result.png')
Predicted Tc (Prob=0.5) = 0.82357
```
input file: `model.result.npy` | output files: `model.result.npy`, `model.result.png`

### Optimize feature map size
After this part, I will explain how to process and get numbers because you need to make pipeline to gather all datas.
Until this step, we obtained single data of phase transition density for a certain model (2x2x2 feature size for 1st CNN layer).

To optimize feature map size, you need to vary feature sizes from 2 to the half of #grid.
Then, you will get following table for N=2048 systems when we use 13 grids:

| feature size  | Phase transition density |
| ------------- | ------------- |
| 2 | 0.80697 |
| 3 | 0.80231 |
| 4 | 0.79924 |
| 5 | 0.80433 |
| 6 | 0.81432 |

All numbers are averaged by 100 random initialization on our ML models.
Because of U-shape of phase transition densities w.r.t feature map size, optimal feature size is chosen by minimum transition density.

### Extrpolate phase transition point using finite-size scaling 

Now, we need to consider finite size effect on phase behavior.
Think about real separation happends in microscopic system, but out simulation system is too tiny to simulate, which gives bias on predicted transition point.
To correct this, we need to extrapolate phase transition for an imaginary system with infinity particles.

For example, following tabls is a result of 0.125 concentration of A with different system size: 
| # particles in system  | Phase transition density |
| ------------- | ------------- |
| 1024 | 0.80283	|
| 2048 | 0.79924	|
| 4069 | 0.82926	|
| 8192 | 0.83698 |

Following finite-size scaling equation, `N^(-1/(3*v))` where v = 0.63012 for the critical exponent of Ising lattice model,
 you can get the number I reported in article: 0.8523 transition point for 0.125 concentration.

Now, you are ready to make data and train ML models for different concentrations to get the critical point.

# Welcome to the machine learning world in Physics!
