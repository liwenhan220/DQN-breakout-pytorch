import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d




data = np.load('train_rewards.npy', allow_pickle=True)
data = gaussian_filter1d(data, sigma=10)
plt.xlim(0, len(data))
plt.plot(np.arange(len(data)), data)
plt.show()
