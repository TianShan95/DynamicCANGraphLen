# import pandas as pd
import numpy as np

# data1 = [[1,0.196699884,-0.077219634],[0.196699884,1,-0.668343535],[-0.077219634,-0.668343535,1]]
# data1 = [1,0.196699884,-0.077219634]
# data2 = [0.196699884,1,-0.668343535]
# data3 = [-0.077219634,-0.668343535,1]
#
# # a = np.round(data1, 2)
# print(a)

states_np = np.array([])
a = np.array([1, 2, 3])
print(np.concatenate((states_np, a), axis=0))