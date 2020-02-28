# CNN_WR
Convolutional Neural Network prediction for the phase behavior of three-dimensional Widom-Rowlinson (WR) Model

# Tutorial from A to Z
## A. generate an initial coordinate
### move to the tutorial folder 

> cd tutorial

### run generator for initial coordinate of random configuration of WR particles;
 in this example, the program tries to insert 512 A particles and 512 B particles (radius = 1*sigma) 
 to satisfy number density 0.5 up to 1,000,000 insertion attempts

output text file, init.ic, will be used for Monte Carlo simulation inpute file

> python ../init_mc/gen_init.py -na 512 -nb 512 -d 0.55 -r 1 -mt 100000
> mv init.ic composite.ic

You can see the text file, composite.ic, for initial coordinate.

Note that positions should be saved in double-precision for accuracy. 

### Run Monte Carlo (MC) Simulation to equilibrate system

For the Fortran 95 version of the program, take a look at README file in ../init_mc/mcrun_v2.tar

> ../init_mc/mcrun.x

intput files: composite.ic, mcrun.inp | output files: composite.tmp, composite.fc, conf.gro, confout.gro, mcrun.out 

While we run MC with double precision coordinates, 
 we take equilibrium system with single-precision file, confout.gro, for next step.

## B. Grid interpolation
### Generate SuperCell
We are going to interpolate 3D coordinates of WR into 3D grid matrix (or lattice, 3D image).
Currently, the built-in grid interpolation function in Python Scipy library has NaN issue on edge interpolation.
To aovid this, we need a super cell with a factor of 2 on axis or (2 x 2 x 2) with copied coordinates.

When applying grid-interpolation on super cell, we remove edges off, then only 1 x 1 x 1 cell 3d image will be left.
In details of method, please see my article. 
For convenience, I upload super.gro file for the supercell coordinate.

### Run grid-interpolation program;

For this case, we will assign 3d coordinates on 10 x 10 x 10 grids (to be close to total number of particles, N=1024),
 and augment the 3d image by a factor of 100 using transformations (translation, replacement of axis, and flipping, etc).

> python ../grid/interpolate.py -i super.gro -g 10 -s 2 -itp three-states -n_ensembles 100 -o grid

input file: super.gro | output file: grid.npy -> size (100 x 10 x 10x 10)

## C. Convolutional Neural Network (CNN) to predict phase transition points

### Make dataset as ML inputs

To do find phase boundary at a fixed concentration, you need to manipulate many grid.npy files at different densities. 
When you prepare those grid.${idx}.npy files with a variety of density, 
 make a list of files in target.list which has two columns; density, grid file index (grid.$idx.npy)
Then, make dataset files for training, test, and evaluation data, shuffling with random seed 1985.

The Training dataset should contain 3d images at two densities; 0.5 and 1.0. 
Rest of 3d images at other densities will be in evaluation file.

> python ~/new_WR/script/machine/block.py -i target.list -ipf grid -s1 0.5 -s2 1.0 -prop 0.0 -nb 1 -seed 1985 -ng 10 -nbe 1 -ne 1

input file: target.list, grid.0.npy, grid.1.npy, ..., grid.5100.npy | output file: train.0.cat.npy, train.0.coord.npy, train.0.temp.npy, eval.0.coord.npy, eval.0.temp.npy

For the sake of time, I upload my dataset on Google Drive for while.
In this tutorial, we are going to use datasets at 0.125 concentration with total 2048 particles (e.g. conc0.125_n2048 folder)  

https://drive.google.com/drive/folders/12hfoGuFf3DwGLALJ9O-m7221JZTT7ozE?usp=sharing

### CNN network model training

