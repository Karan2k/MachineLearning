from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
# uncomment the below line to visualize the plot
# from matplotlib import pyplot as plt 
import numpy as np

hoursCharged = []
batteryLasted = []
showPlot = False # set this variable as true if you want to visualize the dataset

with open("trainingdata.txt", "r") as trainingset:
	for line in trainingset:
		x, y = map(float, line.replace('\n', '').split(','))
		
		hoursCharged.append(x)
		batteryLasted.append(y)

hoursCharged = np.array(hoursCharged)
batteryLasted = np.array(batteryLasted)

if showPlot:
	plt.scatter(hoursCharged, batteryLasted)
	plt.xlabel("Hours Charged")
	plt.ylabel("Battery Lasted (in hours)")
	plt.grid()
	plt.show()

# As it can be observed from the plot, laptop's max battery backup is 8 hours and it gets full charged in 4 hours so we'll only consider the cases
# when laptop is charged for less than 4 hours else we know that the answer would be 8 hours
isFullCharged = (hoursCharged > 4.0)
hoursCharged = hoursCharged[~isFullCharged]
batteryLasted = batteryLasted[~isFullCharged]

hoursCharged = hoursCharged.reshape(hoursCharged.size, 1)
batteryLasted = batteryLasted.reshape(batteryLasted.size, 1)

clf = linear_model.LinearRegression()
clf.fit(hoursCharged, batteryLasted)

testHoursCharged = float(raw_input())

if testHoursCharged > 4.0:
	print 8.0
else:
	print clf.predict(testHoursCharged)[0][0]