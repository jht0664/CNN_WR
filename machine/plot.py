#!/usr/bin/env python3
# fit for Tc and make plot for a machine learning output
#############
# ver 0.1 - coding python by Hyuntae Jung on 2/16/2019
import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='fit sigmoid fn for Tc prediction and make a plot')
## args
parser.add_argument('-i', '--input', default='result.npy', nargs='?',  
	help='input .npy file from machine2.py')
parser.add_argument('-crit', '--criteria', default=0.5, nargs='?', type=float,  
	help='criteria to predict Tc (0.5 means 50:50 into mixed/separated states at the temp.')
parser.add_argument('-o', '--output', default='plot.png', nargs='?',
	help='result plot filename')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
# read argsru
args = parser.parse_args()
# check args
print(" input arguments: {0}".format(args))

# import module
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

# load input file
data_in = np.load(args.input)
plt_temp, plt_cat_mean, plt_cat_std = data_in.T

## make fitting plot with eval. data at different temperatures. 
# The point at which the prediction passes through 0.5 
#  where the critical temperature is located. 
# Thus, we fit a sigmoid to the points and use this to estimate the crossing point (`Tc_fit`)
def sigmoid(x, x0, k):
	return 1 / (1 + 1*np.exp(-k*(x-x0)))

# fit
popt, pcov = curve_fit(sigmoid, plt_temp, plt_cat_mean)
xfit = np.linspace(np.min(plt_temp),np.max(plt_temp),200)
yfit = sigmoid(xfit, *popt)
# and figure out for what T the fit line is closest to 0.5
#avg_pred = (np.min(yfit) + np.max(yfit))/2.
avg_pred = args.criteria
Tc_fit = xfit[np.argmin(np.abs(yfit - avg_pred))]
print("Predicted Tc (Prob={}) = {}".format(avg_pred,str(round(Tc_fit,5))))

## save fitting image
#fig, ax = plt.subplots(nrows=1, ncols=1)
#ax.errorbar(plt_temp[plt_temp < Tc_fit], plt_cat_mean[plt_temp < Tc_fit], 
#	yerr=plt_cat_std[plt_temp < Tc_fit], fmt="bo", label="$Density<Density_{{crit.}}$")
#ax.errorbar(plt_temp[plt_temp > Tc_fit], plt_cat_mean[plt_temp > Tc_fit], 
#	yerr=plt_cat_std[plt_temp > Tc_fit], fmt="ro", label="$Density>Density_{{crit.}}$")
#ax.plot(xfit,yfit, "k-", label="sigmoid fit")
#ax.plot([Tc_fit,Tc_fit],[0.,1.0],'--', color="dodgerblue",linewidth=2.0, 
#	label="$Density_{{crit.,\mathrm{{fit}}}}$ ({:.4f})".format(Tc_fit))
#ax.set_xlabel("$Density$")
#ax.set_ylabel("Probability")
#ax.set_ylim(0,1)
#ax.annotate("de-mixed",xy=(0,0), xytext=(-1.1,1))
#ax.annotate("mixed",xy=(0,0), xytext=(-1,0))
#ax.legend(loc="center right", numpoints=1)
#ax.grid(True)
#plt.savefig(args.output)
#print("Done: save plot in {}".format(args.output))
