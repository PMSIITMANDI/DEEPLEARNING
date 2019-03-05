import numpy as np
x = numpy.arange(5)
x12=numpy.tile(x, (2,2))
x=numpy.mean(x12)
x12=np.copy(x12[1:2,1:2])
print(x)
print (x12)