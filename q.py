import numpy as np
import matplotlib.pyplot as plt

file_path1 = "predicted_volume.npy"

arr = np.load(file_path1)

plt.imshow(arr[:, :, 64])
plt.show()

