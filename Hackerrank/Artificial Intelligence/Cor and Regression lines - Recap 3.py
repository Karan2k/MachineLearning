import numpy as np
from sklearn.linear_model import LinearRegression

physicsScore = np.array([15, 12, 8, 8, 7, 7, 7, 6, 5, 3])
historyScore = np.array([10, 25, 17, 11, 13, 17, 20, 13, 9, 15])

physicsScore = physicsScore.reshape(physicsScore.size, 1)
historyScore = historyScore.reshape(historyScore.size, 1)

clf = LinearRegression()
clf.fit(physicsScore, historyScore)

print clf.predict(10)[0][0]