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
plt_temp, plt_cat1_mean, plt_cat1_std, plt_cat2_mean, plt_cat2_std, plt_cat3_mean, plt_cat3_std = data_in.T

## make fitting plot with eval. data at different temperatures. 
# The point at which the prediction passes through 0.5 
#  where the critical temperature is located. 
# Thus, we fit a sigmoid to the points and use this to estimate the crossing point (`Tc_fit`)
def sigmoid(x, x0, k):
	return 1 / (1 + 1*np.exp(-k*(x-x0)))

# fit
popt1, pcov = curve_fit(sigmoid, plt_temp, plt_cat1_mean)
popt2, pcov = curve_fit(sigmoid, plt_temp, plt_cat2_mean)
#popt3, pcov = curve_fit(sigmoid, plt_temp, plt_cat3_mean)

xfit = np.linspace(np.min(plt_temp),np.max(plt_temp),200)
yfit1 = sigmoid(xfit, *popt1)
yfit2 = sigmoid(xfit, *popt2)
#yfit3 = sigmoid(xfit, *popt3)
# and figure out for what T the fit line is closest to 0.5
#avg_pred = (np.min(yfit) + np.max(yfit))/2.
avg_pred = args.criteria
Tc1_fit = xfit[np.argmin(np.abs(yfit1 - avg_pred))]
Tc2_fit = xfit[np.argmin(np.abs(yfit2 - avg_pred))]
Tc3_fit = plt_temp[np.argmax(plt_cat3_mean)]
print("Predicted Tc1 (Prob={}) = {}".format(avg_pred,str(round(Tc1_fit,5))))
print("Predicted Tc2 (Prob={}) = {}".format(avg_pred,str(round(Tc2_fit,5))))
print("Predicted Tc3 (max) = {}".format(str(round(Tc3_fit,5))))

## save fitting image
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
#ax3.plot(xfit,yfit3, "k-", label="sigmoid fit3")
ax1.plot([Tc1_fit,Tc1_fit],[0.,1.0],'--', color="dodgerblue",linewidth=2.0, 
	label="fit1_$Density_{{crit.,\mathrm{{fit}}}}$ ({:.4f})".format(Tc1_fit))
ax2.plot([Tc2_fit,Tc2_fit],[0.,1.0],'--', color="dodgerblue",linewidth=2.0, 
	label="fit2_$Density_{{crit.,\mathrm{{fit}}}}$ ({:.4f})".format(Tc2_fit))
ax3.plot([Tc3_fit,Tc3_fit],[0.,1.0],'--', color="dodgerblue",linewidth=2.0, 
	label="fit3_$Density_{{crit.,\mathrm{{fit}}}}$ ({:.4f})".format(Tc3_fit))
# overlay on fit values
ax1.errorbar(plt_temp[plt_temp < Tc1_fit], plt_cat1_mean[plt_temp < Tc1_fit], 
	yerr=plt_cat1_std[plt_temp < Tc1_fit], fmt="bo", label="fit1_$Density<Density_{{crit.}}$")
ax1.errorbar(plt_temp[plt_temp > Tc1_fit], plt_cat1_mean[plt_temp > Tc1_fit], 
	yerr=plt_cat1_std[plt_temp > Tc1_fit], fmt="ro", label="fit1_$Density>Density_{{crit.}}$")
ax2.errorbar(plt_temp[plt_temp < Tc2_fit], plt_cat2_mean[plt_temp < Tc2_fit], 
	yerr=plt_cat2_std[plt_temp < Tc2_fit], fmt="bo", label="fit2_$Density<Density_{{crit.}}$")
ax2.errorbar(plt_temp[plt_temp > Tc2_fit], plt_cat2_mean[plt_temp > Tc2_fit], 
	yerr=plt_cat2_std[plt_temp > Tc2_fit], fmt="ro", label="fit2_$Density>Density_{{crit.}}$")
ax3.errorbar(plt_temp[plt_temp < Tc3_fit], plt_cat3_mean[plt_temp < Tc3_fit], 
	yerr=plt_cat3_std[plt_temp < Tc3_fit], fmt="bo", label="fit3_$Density<Density_{{crit.}}$")
ax3.errorbar(plt_temp[plt_temp > Tc3_fit], plt_cat3_mean[plt_temp > Tc3_fit], 
	yerr=plt_cat3_std[plt_temp > Tc3_fit], fmt="ro", label="fit3_$Density>Density_{{crit.}}$")
ax1.plot(xfit,yfit1, "k-", label="sigmoid fit1")
ax2.plot(xfit,yfit2, "k-", label="sigmoid fit2")
ax3.set_xlabel("$Density$")
ax1.set_ylabel("Probability")
ax2.set_ylabel("Probability")
ax3.set_ylabel("Probability")
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)
ax3.set_ylim(0,1)
#ax.annotate("de-mixed",xy=(0,0), xytext=(-1.1,1))
#ax.annotate("mixed",xy=(0,0), xytext=(-1,0))
#ax.legend(loc="center right", numpoints=1)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
plt.savefig(args.output)
print("Done: save plot in {}".format(args.output))
