import numpy as np
import scipy as sp

physicsScore = np.array([15, 12, 8, 8, 7, 7, 7, 6, 5, 3])
historyScore = np.array([10, 25, 17, 11, 13, 17, 20, 13, 9, 15])

model = sp.polyfit(physicsScore, historyScore, 1)

# index 0 stores the slope and index 1 stores the intercept
print model[0]