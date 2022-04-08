import pandas as pd
import numpy as np
import scipy
import scipy.linalg

data1 = [[1,0.196699884,-0.077219634],[0.196699884,1,-0.668343535],[-0.077219634,-0.668343535,1]]

# a = pd.DataFrame(np.round(data1, 2))
# print(a)

a, b = scipy.linalg.eigh(data1)
a1, b1 = np.linalg.eigh(data1)
print(f'type(a) : {type(a)}')
print(f'a : {a}')
print(f'type(a1) : {type(a1)}')
print(f'a : {a1}')
print(f'type(b) : {type(b)}')
print(f'b : {b}')
print(f'type(b1) : {type(b1)}')
print(f'b : {b1}')