import numpy as np
import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='remove odd data for 3c beyond some criteria')
## args
parser.add_argument('-i', '--input', default='fit2.value', nargs='?', 
	help='input list file of transition densities/temperatures')
parser.add_argument('-t1', '--temp1', default=0.5, nargs='?', type=float, 
	help='lowest temperature/density1')
parser.add_argument('-t2', '--temp2', default=1.0, nargs='?', type=float, 
	help='highest temperature/density2')
parser.add_argument('-c', '--crit', default=0.02, nargs='?', type=float,
        help='acceptance deviation of transition temperature')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
# read args
args = parser.parse_args()
# check args
print(" input arguments: {0}".format(args))

list_val=np.loadtxt(args.input)
list_val=list_val.reshape(-1,3)
b1=np.array([])
b2=np.array([])
b3=np.array([])
for iline in list_val:
	if (iline[0] <= iline[2]) and (iline[1] >= iline[2]):
		if (iline[0] > args.temp1+args.crit) and (iline[1] < args.temp2-args.crit):
			b1=np.append(b1,iline[0])
			b2=np.append(b2,iline[1])
			b3=np.append(b3,iline[2])

print("b1 = {:.5f} {:.5f}".format(np.average(b1),np.std(b1)))
print("b2 = {:.5f} {:.5f}".format(np.average(b2),np.std(b2)))
print("b3 = {:.5f} {:.5f}".format(np.average(b3),np.std(b3)))
