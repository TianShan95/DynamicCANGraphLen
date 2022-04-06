import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000, 20)
ax = sns.heatmap(data, xticklabels=10, yticklabels=False)
plt.show()
