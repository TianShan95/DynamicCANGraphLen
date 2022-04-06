import numpy as np
import pandas as pd
import testseaborn as sns
import matplotlib.pyplot as plt


data1 = [[1,0.196699884,-1],[0.196699884,1,-2],[-0.077219634,-0.668343535,-3]]
fig, ax = plt.subplots(figsize=(50, 10))
key_list = ['A', 'B', 'C']
key_list1 = ['A1', 'B1', 'C1']
sns.heatmap(pd.DataFrame(np.round(data1, 2), columns=key_list, index=key_list1), annot=True, vmax=1, vmin=-1,
            square=True, cmap="YlGnBu")
# ax.set_title('', fontsize=18)
plt.show()
plt.savefig()
