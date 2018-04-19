import numpy as np
import matplotlib.pyplot as plt

with np.load('images/im1/depth.npz') as data:
    plt.figure()
    plt.imshow(data['d_img'])
    plt.show()

