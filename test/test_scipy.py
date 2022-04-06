import pandas as pd
import test_numpy as np

data1 = [[1,0.196699884,-0.077219634],[0.196699884,1,-0.668343535],[-0.077219634,-0.668343535,1]]

a = pd.DataFrame(np.round(data1, 2))
print(a)