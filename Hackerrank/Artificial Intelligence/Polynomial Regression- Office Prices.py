# Enter your code here. Read input from STDIN. Print output to STDOUT
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

numberOfFeatures, trainingDataSetSize = map(int, raw_input().split())

features = []
prices = []

for i in xrange(trainingDataSetSize):
    tempFeature = map(float, raw_input().split())
    feature = []
    
    for j in xrange(numberOfFeatures):
        feature.append(tempFeature[j])
    features.append(feature)
    
    prices.append(tempFeature[numberOfFeatures])
           
testDataSize = int(raw_input())

poly = PolynomialFeatures(degree=3)
transformedFeatures = poly.fit_transform(features)
clf = linear_model.LinearRegression()
clf.fit(transformedFeatures, prices)

for i in xrange(testDataSize):
    predict = map(float, raw_input().split())
    
    prediction = poly.fit_transform(predict)
    print clf.predict(prediction)[0]