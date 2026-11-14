import numpy as np

data = np.load(r'/home/karunya/tapeinos/resources/ca_44e9c6/homography_points.npz')

for key in data:
    np.savetxt(f"{key}.csv", data[key], delimiter=',')