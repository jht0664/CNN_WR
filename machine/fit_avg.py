import numpy as np
list_val=np.loadtxt('fit2.value')
b=np.array([])
for i in np.arange(len(list_val)):
	if list_val[i] < 0.52:
		b=np.append(b,i)
	if list_val[i] > 0.98:
		b=np.append(b,i)

list_val=np.delete(list_val,b)

print("{:.5f} {:.5f}".format(np.average(list_val),np.std(list_val)))
