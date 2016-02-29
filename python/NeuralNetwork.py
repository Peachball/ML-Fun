import theano
import numpy as np

class FFNetwork:
	def __init__(self, dimensions):
		weights = np.rand(dimensions[0], dimensions[1])
		for i in dimensions:

