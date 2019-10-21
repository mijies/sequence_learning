import sys
from common.config import GPU

if GPU:
	import cupy as np
	np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    # numpy.add.at and cupy.scatter_add play the same role.
    # Performs unbuffered in place operation on operand a for elements specified by indices.
	np.add.at = np.scatter_add  

	print("  GPU mode(cupy)")

else:
	import numpy as np


def to_cpu(x):
	import numpy # not common.np
	if type(x) == numpy.ndarray:
		return x
	return np.asnumpy(x)
	

def to_gpu(x):
	import cupy
	if type(x) == cupy.ndarray:
		return x
	return cupy.asarray(x)
