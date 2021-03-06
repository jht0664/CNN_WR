import numpy as np
import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='remove odd data beyond criteria')
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
b=np.array([])
for i in np.arange(len(list_val)):
	if list_val[i] <= args.temp1+args.crit:
		b=np.append(b,i)
	if list_val[i] >= args.temp2-args.crit:
		b=np.append(b,i)

list_val=np.delete(list_val,b)

print("{:.5f} {:.5f}".format(np.average(list_val),np.std(list_val)))
